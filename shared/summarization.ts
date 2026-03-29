// =============================================================================
// kivv - Two-Stage AI Summarization Client
// =============================================================================
// Supports both Anthropic Claude and MiniMax
// Stage 1: Fast model for relevance triage (0.0-1.0 score)
// Stage 2: Strong model for detailed summaries (only if score >= threshold)
// Cost optimization: ~96% savings on irrelevant papers
// Rate limiting: Configurable per provider
// Budget tracking: Circuit breaker at $1/day
// =============================================================================

import { hashContent } from './utils';
import {
  AI_PROVIDER,
  MINIMAX_API_BASE_URL,
  MINIMAX_MODEL,
  MINIMAX_RATE_LIMIT_MS,
  MINIMAX_JITTER_MIN_MS,
  MINIMAX_JITTER_MAX_MS,
  CLAUDE_HAIKU_MODEL,
  CLAUDE_SONNET_MODEL,
  MAX_SUMMARY_OUTPUT_TOKENS,
  MAX_TRIAGE_OUTPUT_TOKENS,
  DEFAULT_RELEVANCE_THRESHOLD,
  ANTHROPIC_RATE_LIMIT_MS,
  ANTHROPIC_JITTER_MIN_MS,
  ANTHROPIC_JITTER_MAX_MS,
  DAILY_BUDGET_CAP_USD,
  ANTHROPIC_API_BASE_URL,
} from './constants';

// =============================================================================
// Types & Interfaces
// =============================================================================

/**
 * Two-stage summarization result
 */
export interface SummarizationResult {
  /** Generated summary (null if irrelevant/skipped/error) */
  summary: string | null;
  /** Relevance score from triage (0.0-1.0) */
  relevance_score: number;
  /** SHA-256 hash of title + abstract for deduplication */
  content_hash: string;
  /** Cost of triage in USD */
  haiku_cost: number;
  /** Cost of summary in USD */
  sonnet_cost: number;
  /** Total cost (haiku + sonnet) in USD */
  total_cost: number;
  /** Reason paper was skipped (if applicable) */
  skipped_reason?: 'irrelevant' | 'budget_exceeded' | 'error';
}

/**
 * Anthropic API response structure
 */
interface AnthropicResponse {
  id: string;
  type: string;
  role: string;
  content: Array<{
    type: string;
    text: string;
  }>;
  model: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

/**
 * OpenAI/MiniMax compatible API response
 */
interface OpenAIResponse {
  id: string;
  model: string;
  choices: Array<{
    message: {
      role: string;
      content: string;
      reasoning_content?: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// =============================================================================
// Summarization Client
// =============================================================================

/**
 * Two-stage AI summarization client supporting multiple providers
 *
 * Stage 1: Fast model for relevance triage (~$0.0001/paper)
 * Stage 2: Strong model for detailed summaries (~$0.003/paper)
 *
 * Features:
 * - Multi-provider support: Anthropic Claude or MiniMax
 * - Rate limiting with jitter
 * - Budget tracking: Circuit breaker at $1/day
 * - Content hashing: Detect duplicate papers
 * - Error handling: Graceful failures with retry
 */
export class SummarizationClient {
  private apiKey: string;
  private provider: 'anthropic' | 'minimax';
  private lastRequestTime = 0;
  private totalCost = 0;

  /**
   * Create a new summarization client
   *
   * @param apiKey - API key (Anthropic or MiniMax)
   * @param provider - AI provider ('anthropic' or 'minimax')
   */
  constructor(apiKey: string, provider: 'anthropic' | 'minimax' = 'minimax') {
    this.apiKey = apiKey;
    this.provider = provider;
  }

  // ===========================================================================
  // Rate Limiting
  // ===========================================================================

  /**
   * Enforce rate limit based on provider
   */
  private async enforceRateLimit(): Promise<void> {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;

    let rateLimitMs: number;
    let jitterMin: number;
    let jitterMax: number;

    if (this.provider === 'minimax') {
      rateLimitMs = MINIMAX_RATE_LIMIT_MS;
      jitterMin = MINIMAX_JITTER_MIN_MS;
      jitterMax = MINIMAX_JITTER_MAX_MS;
    } else {
      rateLimitMs = ANTHROPIC_RATE_LIMIT_MS;
      jitterMin = ANTHROPIC_JITTER_MIN_MS;
      jitterMax = ANTHROPIC_JITTER_MAX_MS;
    }

    const jitter = Math.random() * (jitterMax - jitterMin) + jitterMin;
    const requiredDelay = rateLimitMs + jitter;

    if (timeSinceLastRequest < requiredDelay) {
      const sleepMs = requiredDelay - timeSinceLastRequest;
      await new Promise((resolve) => setTimeout(resolve, sleepMs));
    }

    this.lastRequestTime = Date.now();
  }

  // ===========================================================================
  // Stage 1: Triage
  // ===========================================================================

  /**
   * Stage 1: Use fast model to quickly assess paper relevance
   */
  private async triageRelevance(
    title: string,
    abstract: string,
    userTopics: string[]
  ): Promise<{ score: number; cost: number }> {
    await this.enforceRateLimit();

    const topicList = userTopics.join(', ');

    const prompt = `You are evaluating research papers for relevance to the user's research interests.

USER INTERESTS: ${topicList}

SCORING CRITERIA:
- 0.9-1.0: Highly relevant, directly addresses user's interests
- 0.7-0.9: Relevant, closely related to user's field
- 0.5-0.7: Somewhat relevant, tangential connection
- 0.3-0.5: Low relevance, indirect connection
- 0.0-0.3: Not relevant, different field entirely

Paper Title: ${title}

Abstract: ${abstract}

Return ONLY a number between 0.0 and 1.0. No explanation.`;

    if (this.provider === 'minimax') {
      const response = await this.callMiniMax(MINIMAX_MODEL, prompt, MAX_TRIAGE_OUTPUT_TOKENS);
      // M2.5 puts responses in reasoning_content, check both fields
      const message = response.choices[0]?.message;
      const scoreText = (message?.content?.trim() || message?.reasoning_content?.trim() || '0.5');
      const score = parseFloat(scoreText);

      if (isNaN(score) || score < 0 || score > 1) {
        return { score: 0.5, cost: this.calculateCost(response.usage, 'triage') };
      }

      return { score, cost: this.calculateCost(response.usage, 'triage') };
    } else {
      const response = await this.callClaude(CLAUDE_HAIKU_MODEL, prompt, MAX_TRIAGE_OUTPUT_TOKENS);
      const scoreText = response.content[0].text.trim();
      const score = parseFloat(scoreText);

      if (isNaN(score) || score < 0 || score > 1) {
        return { score: 0.5, cost: this.calculateCost(response.usage, 'triage') };
      }

      return { score, cost: this.calculateCost(response.usage, 'triage') };
    }
  }

  // ===========================================================================
  // Stage 2: Summary
  // ===========================================================================

  /**
   * Stage 2: Use strong model to generate detailed summary
   */
  private async generateSummary(
    title: string,
    abstract: string
  ): Promise<{ summary: string; cost: number }> {
    await this.enforceRateLimit();

    const prompt = `Write a comprehensive summary of this research paper in English ONLY. Include:
1. The problem being addressed
2. The approach or method used
3. The key results or findings
4. Any notable innovations or contributions

IMPORTANT:
- Write the entire summary in English. Do NOT use any Chinese characters.
- ONLY wrap TRUE mathematical expressions in $$...$$. Examples of TRUE math:
  - Equations with operators: $$O(N^3)$$, $$\\nabla f(x)$$, $$\\frac{d}{dx}$$
  - Greek letters in formulas: $$\\alpha$$, $$\\beta$$, $$\\lambda$$
  - Matrix/vector notation: $$\\mathbf{W}$$, $$\\vec{x}$$
  - Statistical symbols: $$p\\text{-value}$$, $$\\mathbb{E}[X]$$
- DO NOT wrap these in $$:
  - Acronyms like LLMs, RAG, RL, NLP, CNN, GPT, LLM, AI, ML
  - Regular variable names like "function f" (just write "function f", not "$$f$$")
  - Plain words like "matrix A" (just write "matrix A")
- Write acronyms and abbreviations as plain text, NOT as math formulas

Paper Title: ${title}

Abstract: ${abstract}

Provide a detailed summary in 2-4 paragraphs. Be informative and capture the essential contributions.`;

    if (this.provider === 'minimax') {
      const response = await this.callMiniMax(MINIMAX_MODEL, prompt, MAX_SUMMARY_OUTPUT_TOKENS);
      const message = response.choices[0]?.message;
      // M2.5 puts responses in reasoning_content, check both fields
      const summary = (message?.content?.trim() || message?.reasoning_content?.trim() || '');
      return { summary, cost: this.calculateCost(response.usage, 'summary') };
    } else {
      const response = await this.callClaude(CLAUDE_SONNET_MODEL, prompt, MAX_SUMMARY_OUTPUT_TOKENS);
      const summary = response.content[0].text.trim();
      return { summary, cost: this.calculateCost(response.usage, 'summary') };
    }
  }

  // ===========================================================================
  // Two-Stage Pipeline
  // ===========================================================================

  /**
   * Execute two-stage summarization pipeline
   */
  async summarize(
    title: string,
    abstract: string,
    userTopics: string[],
    relevanceThreshold = DEFAULT_RELEVANCE_THRESHOLD,
    currentTotalCost = 0
  ): Promise<SummarizationResult> {
    if (currentTotalCost >= DAILY_BUDGET_CAP_USD) {
      return {
        summary: null,
        relevance_score: 0,
        content_hash: await hashContent(title + abstract),
        haiku_cost: 0,
        sonnet_cost: 0,
        total_cost: 0,
        skipped_reason: 'budget_exceeded',
      };
    }

    const content_hash = await hashContent(title + abstract);

    try {
      // Stage 1: Triage
      const { score, cost: haikuCost } = await this.triageRelevance(title, abstract, userTopics);
      this.totalCost += haikuCost;

      // Check relevance threshold
      if (score < relevanceThreshold) {
        return {
          summary: null,
          relevance_score: score,
          content_hash,
          haiku_cost: haikuCost,
          sonnet_cost: 0,
          total_cost: haikuCost,
          skipped_reason: 'irrelevant',
        };
      }

      // Stage 2: Summary (only for relevant papers)
      const { summary, cost: sonnetCost } = await this.generateSummary(title, abstract);
      this.totalCost += sonnetCost;

      return {
        summary,
        relevance_score: score,
        content_hash,
        haiku_cost: haikuCost,
        sonnet_cost: sonnetCost,
        total_cost: haikuCost + sonnetCost,
      };
    } catch (error) {
      console.error('Summarization failed:', error);
      return {
        summary: null,
        relevance_score: 0,
        content_hash,
        haiku_cost: 0,
        sonnet_cost: 0,
        total_cost: 0,
        skipped_reason: 'error',
      };
    }
  }

  // ===========================================================================
  // API Clients
  // ===========================================================================

  /**
   * Call MiniMax/OpenAI compatible API
   */
  private async callMiniMax(
    model: string,
    prompt: string,
    maxTokens: number
  ): Promise<OpenAIResponse> {
    const response = await fetch(`${MINIMAX_API_BASE_URL}/text/chatcompletion_v2`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        max_tokens: maxTokens,
        messages: [
          {
            role: 'user',
            content: prompt,
          },
        ],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`MiniMax API error: ${response.status} - ${errorText}`);
    }

    return (await response.json()) as OpenAIResponse;
  }

  /**
   * Call Anthropic Claude API
   */
  private async callClaude(
    model: string,
    prompt: string,
    maxTokens: number
  ): Promise<AnthropicResponse> {
    const response = await fetch(`${ANTHROPIC_API_BASE_URL}/messages`, {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model,
        max_tokens: maxTokens,
        messages: [
          {
            role: 'user',
            content: prompt,
          },
        ],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Anthropic API error: ${response.status} - ${errorText}`);
    }

    return (await response.json()) as AnthropicResponse;
  }

  // ===========================================================================
  // Cost Calculation
  // ===========================================================================

  /**
   * Calculate cost based on token usage and model pricing
   *
   * MiniMax M2.2 pricing (approximate):
   * - Input: $0.10 per 1M tokens
   * - Output: $0.10 per 1M tokens
   *
   * Claude pricing:
   * - Haiku: $0.25/$1.25 per 1M tokens
   * - Sonnet: $3.00/$15.00 per 1M tokens
   */
  private calculateCost(
    usage: { input_tokens?: number; output_tokens?: number; prompt_tokens?: number; completion_tokens?: number },
    type: 'triage' | 'summary'
  ): number {
    // Handle both Anthropic and OpenAI format
    const inputTokens = usage.input_tokens || usage.prompt_tokens || 0;
    const outputTokens = usage.output_tokens || usage.completion_tokens || 0;

    if (this.provider === 'minimax') {
      // MiniMax pricing: ~$0.10/1M input and output
      const rate = type === 'triage' ? 0.10 : 0.10;
      return (inputTokens + outputTokens) * rate / 1_000_000;
    } else {
      // Claude pricing
      if (type === 'triage') {
        // Haiku
        return inputTokens * 0.25 / 1_000_000 + outputTokens * 1.25 / 1_000_000;
      } else {
        // Sonnet
        return inputTokens * 3.0 / 1_000_000 + outputTokens * 15.0 / 1_000_000;
      }
    }
  }

  // ===========================================================================
  // Budget Tracking
  // ===========================================================================

  getTotalCost(): number {
    return this.totalCost;
  }

  resetCost(): void {
    this.totalCost = 0;
  }

  isBudgetExceeded(): boolean {
    return this.totalCost >= DAILY_BUDGET_CAP_USD;
  }

  getRemainingBudget(): number {
    return Math.max(0, DAILY_BUDGET_CAP_USD - this.totalCost);
  }
}
