// =============================================================================
// kivv - Shared Constants
// =============================================================================
// Configuration constants used across MCP server and automation workers
// All values match PRD specifications exactly
// =============================================================================
// =============================================================================
// arXiv API Configuration
// =============================================================================
/** arXiv API base URL for querying papers */
export const ARXIV_API_BASE_URL = 'http://export.arxiv.org/api/query';
/** arXiv rate limit: 1 request per interval */
export const ARXIV_RATE_LIMIT_REQUESTS = 1;
/** arXiv rate limit interval: 3 seconds between requests */
export const ARXIV_RATE_LIMIT_INTERVAL_MS = 3000;
/** Random jitter added to arXiv requests (0-200ms) to avoid detection patterns */
export const ARXIV_RATE_LIMIT_JITTER_MS = 200;
/** Maximum results per arXiv API request (API limit) */
export const ARXIV_MAX_RESULTS_PER_REQUEST = 100;
// =============================================================================
// Anthropic API Configuration
// =============================================================================
/** Anthropic API base URL */
export const ANTHROPIC_API_BASE_URL = 'https://api.anthropic.com/v1';
/** Anthropic rate limit: 5 requests per second */
export const ANTHROPIC_RATE_LIMIT_REQUESTS = 5;
/** Anthropic rate limit interval: 1 second */
export const ANTHROPIC_RATE_LIMIT_INTERVAL_MS = 1000;
// =============================================================================
// Model Identifiers
// =============================================================================
/** Claude 3.5 Haiku model identifier (for triage) */
export const CLAUDE_HAIKU_MODEL = 'claude-3-5-haiku-20241022';
/** Claude 3.5 Sonnet model identifier (for summaries) */
export const CLAUDE_SONNET_MODEL = 'claude-3-5-sonnet-20241022';
// =============================================================================
// Token Limits
// =============================================================================
/** Maximum output tokens for Sonnet summaries (3 bullet points) */
export const MAX_SUMMARY_OUTPUT_TOKENS = 5000;
/** Maximum output tokens for Haiku triage (just a number 0.0-1.0) */
export const MAX_TRIAGE_OUTPUT_TOKENS = 10;
// =============================================================================
// Cost Tracking & Budget Enforcement
// =============================================================================
/** Daily budget cap in USD - circuit breaker threshold */
export const DAILY_BUDGET_CAP_USD = 1.0;
/** Budget warning threshold (50% of daily cap) */
export const BUDGET_WARNING_THRESHOLD = 0.5;
/** Budget critical threshold (80% of daily cap) */
export const BUDGET_CRITICAL_THRESHOLD = 0.8;
// =============================================================================
// Relevance Thresholds
// =============================================================================
/** Default relevance threshold for paper triage (papers below this are skipped) */
export const DEFAULT_RELEVANCE_THRESHOLD = 0.7;
/** Minimum allowed relevance threshold */
export const MIN_RELEVANCE_THRESHOLD = 0.0;
/** Maximum allowed relevance threshold */
export const MAX_RELEVANCE_THRESHOLD = 1.0;
// =============================================================================
// Cache TTLs (Time To Live in seconds)
// =============================================================================
/** RSS feed cache TTL (5 minutes) */
export const RSS_FEED_CACHE_TTL = 300;
/** Rate limit state TTL (1 hour) */
export const RATE_LIMIT_STATE_TTL = 3600;
/** Checkpoint TTL (24 hours) - for resumable cron jobs */
export const CHECKPOINT_TTL = 86400;
// =============================================================================
// Pagination Defaults
// =============================================================================
/** Default page limit for list_library tool */
export const DEFAULT_PAGE_LIMIT = 50;
/** Maximum page limit (prevent excessive queries) */
export const MAX_PAGE_LIMIT = 100;
// =============================================================================
// HTTP Status Codes
// =============================================================================
/** 200 OK - Request succeeded */
export const HTTP_OK = 200;
/** 201 Created - Resource created successfully */
export const HTTP_CREATED = 201;
/** 400 Bad Request - Invalid request parameters */
export const HTTP_BAD_REQUEST = 400;
/** 401 Unauthorized - Missing or invalid authentication */
export const HTTP_UNAUTHORIZED = 401;
/** 403 Forbidden - Valid auth but insufficient permissions */
export const HTTP_FORBIDDEN = 403;
/** 404 Not Found - Resource does not exist */
export const HTTP_NOT_FOUND = 404;
/** 429 Too Many Requests - Rate limit exceeded */
export const HTTP_TOO_MANY_REQUESTS = 429;
/** 500 Internal Server Error - Server error */
export const HTTP_INTERNAL_ERROR = 500;
// =============================================================================
// Error Codes (for API responses)
// =============================================================================
/** Invalid API key provided */
export const ERROR_INVALID_API_KEY = 'INVALID_API_KEY';
/** User account is inactive */
export const ERROR_USER_INACTIVE = 'USER_INACTIVE';
/** Missing authentication header */
export const ERROR_MISSING_AUTH = 'MISSING_AUTH';
/** Rate limit exceeded */
export const ERROR_RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED';
/** Daily budget exceeded */
export const ERROR_BUDGET_EXCEEDED = 'BUDGET_EXCEEDED';
/** Invalid input parameters */
export const ERROR_INVALID_INPUT = 'INVALID_INPUT';
/** Database error */
export const ERROR_DATABASE = 'DATABASE_ERROR';
/** External API error (arXiv, Anthropic) */
export const ERROR_EXTERNAL_API = 'EXTERNAL_API_ERROR';
