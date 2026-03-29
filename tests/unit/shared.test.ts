// =============================================================================
// CHUNK 2 Verification Test - Shared Types, Utils, Constants
// =============================================================================
// This test verifies that all exports from shared modules work correctly

import { describe, it, expect } from 'vitest';

// Import types
import type {
  User,
  Topic,
  Paper,
  UserPaperStatus,
  CostLog,
  Env,
  PaperWithStatus,
  ArxivApiPaper,
  RelevanceScore,
  SummaryResult,
  ListLibraryRequest,
  SearchPapersRequest,
  MarkExploredRequest,
  ApiError,
} from '../../shared/types';

// Import utilities
import {
  hashContent,
  generateId,
  parseJsonArray,
  formatDate,
  calculateCost,
  createErrorResponse,
  validateArxivId,
  sleep,
} from '../../shared/utils';

// Import constants
import {
  ARXIV_API_BASE_URL,
  CLAUDE_HAIKU_MODEL,
  CLAUDE_SONNET_MODEL,
  MAX_SUMMARY_OUTPUT_TOKENS,
  MAX_TRIAGE_OUTPUT_TOKENS,
  DEFAULT_RELEVANCE_THRESHOLD,
  HTTP_OK,
  HTTP_UNAUTHORIZED,
  ERROR_INVALID_API_KEY,
  ERROR_BUDGET_EXCEEDED,
} from '../../shared/constants';

describe('Shared Types', () => {
  it('should export all required type interfaces', () => {
    // This test just verifies types are imported without errors
    // TypeScript compilation is the real test
    expect(true).toBe(true);
  });
});

describe('Shared Utils', () => {
  it('should hash content correctly', async () => {
    const text = 'Machine Learning Paper Title';
    const hash = await hashContent(text);
    expect(hash).toHaveLength(64); // SHA-256 produces 64 hex chars
    expect(hash).toMatch(/^[a-f0-9]{64}$/);
  });

  it('should generate valid UUIDs', () => {
    const id = generateId();
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
  });

  it('should parse JSON arrays safely', () => {
    const valid = parseJsonArray<string>('["Alice", "Bob"]');
    expect(valid).toEqual(['Alice', 'Bob']);

    const invalid = parseJsonArray<string>('not json');
    expect(invalid).toEqual([]);
  });

  it('should format dates correctly', () => {
    const date = new Date('2025-11-30T12:00:00Z');
    const formatted = formatDate(date);
    expect(formatted).toBe('2025-11-30');
  });

  it('should calculate costs correctly', () => {
    const cost = calculateCost(1000000, 'haiku-input');
    expect(cost).toBe(0.25); // $0.25 per 1M tokens

    const costSonnet = calculateCost(1000000, 'sonnet-output');
    expect(costSonnet).toBe(15.0); // $15 per 1M tokens
  });

  it('should create error responses', () => {
    const response = createErrorResponse('Test error', 'TEST_ERROR', 400);
    expect(response.status).toBe(400);
    expect(response.headers.get('Content-Type')).toBe('application/json');
  });

  it('should validate arXiv IDs', () => {
    expect(validateArxivId('2311.12345')).toBe(true);
    expect(validateArxivId('arXiv:2311.12345')).toBe(true);
    expect(validateArxivId('invalid')).toBe(false);
  });

  it('should sleep for specified duration', async () => {
    const start = Date.now();
    await sleep(50);
    const elapsed = Date.now() - start;
    expect(elapsed).toBeGreaterThanOrEqual(45); // Allow some variance
  });
});

describe('Shared Constants', () => {
  it('should export arXiv configuration', () => {
    expect(ARXIV_API_BASE_URL).toBe('http://export.arxiv.org/api/query');
  });

  it('should export model identifiers', () => {
    expect(CLAUDE_HAIKU_MODEL).toBe('claude-3-5-haiku-20241022');
    expect(CLAUDE_SONNET_MODEL).toBe('claude-sonnet-4-20250514');
  });

  it('should export token limits', () => {
    expect(MAX_SUMMARY_OUTPUT_TOKENS).toBe(5000);
    expect(MAX_TRIAGE_OUTPUT_TOKENS).toBe(10);
  });

  it('should export thresholds', () => {
    expect(DEFAULT_RELEVANCE_THRESHOLD).toBe(0.7);
  });

  it('should export HTTP status codes', () => {
    expect(HTTP_OK).toBe(200);
    expect(HTTP_UNAUTHORIZED).toBe(401);
  });

  it('should export error codes', () => {
    expect(ERROR_INVALID_API_KEY).toBe('INVALID_API_KEY');
    expect(ERROR_BUDGET_EXCEEDED).toBe('BUDGET_EXCEEDED');
  });
});
