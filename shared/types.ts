// =============================================================================
// kivv - Shared TypeScript Types
// =============================================================================
// All interfaces match the D1 database schema exactly
// Used across MCP server and automation workers
// =============================================================================

// =============================================================================
// Database Entity Types
// =============================================================================

/**
 * User entity - Matches users table in D1
 */
export interface User {
  id: number;
  username: string;
  email: string;
  api_key: string;
  display_name: string | null;
  created_at: string;
  last_login: string | null;
  is_active: boolean;
}

/**
 * Topic entity - Matches topics table in D1
 * Represents a user's research topic with arXiv query and settings
 */
export interface Topic {
  id: number;
  user_id: number;
  topic_name: string;
  arxiv_query: string;
  enabled: boolean;
  relevance_threshold: number;
  max_papers_per_day: number;
  generate_summaries: boolean;
  created_at: string;
  last_collection_at: string | null;
  last_cursor: string | null;
}

/**
 * Paper entity - Matches papers table in D1
 * Stores arXiv papers with summaries and metadata
 */
export interface Paper {
  id: number;
  arxiv_id: string;
  title: string;
  authors: string; // JSON array as string
  abstract: string;
  categories: string; // JSON array as string
  published_date: string;
  pdf_url: string;
  r2_key: string | null;
  summary: string | null;
  summary_generated_at: string | null;
  summary_model: string | null;
  relevance_score: number | null;
  content_hash: string | null;
  collected_for_user_id: number | null;
  created_at: string;
}

/**
 * UserPaperStatus entity - Matches user_paper_status table in D1
 * Tracks per-user exploration, bookmarks, and notes for papers
 */
export interface UserPaperStatus {
  user_id: number;
  paper_id: number;
  explored: boolean;
  bookmarked: boolean;
  notes: string | null;
  read_at: string | null;
  created_at: string;
}

/**
 * CostLog entity - Matches cost_logs table in D1
 * Tracks API usage costs for budget enforcement
 */
export interface CostLog {
  id: number;
  date: string; // YYYY-MM-DD format
  user_id: number | null;
  service: string; // 'haiku' | 'sonnet'
  papers_processed: number;
  tokens_input: number;
  tokens_output: number;
  cost_usd: number;
  created_at: string;
}

// =============================================================================
// Cloudflare Workers Environment Bindings
// =============================================================================

/**
 * Environment bindings for Cloudflare Workers
 * Includes D1 database, KV namespace, R2 bucket, and environment variables
 */
export interface Env {
  // Cloudflare bindings
  DB: D1Database;
  CACHE: KVNamespace;
  PAPERS: R2Bucket;

  // Environment variables
  MINIMAX_API_KEY: string;
  D1_DATABASE_ID: string;
  KV_NAMESPACE_ID: string;
  R2_BUCKET_NAME: string;
  NTFY_TOPIC?: string;
}

// =============================================================================
// API Response Types
// =============================================================================

/**
 * Paper with user-specific status fields
 * Used when returning papers to MCP tools
 */
export interface PaperWithStatus extends Paper {
  explored?: boolean;
  bookmarked?: boolean;
  notes?: string | null;
}

/**
 * arXiv API paper format (before DB insertion)
 * Parsed from arXiv Atom XML feed
 */
export interface ArxivApiPaper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  published: string;
  pdf_url: string;
}

/**
 * Relevance score result from Haiku triage
 */
export interface RelevanceScore {
  paper_id: number;
  score: number;
  threshold: number;
  passed: boolean;
}

/**
 * Summary generation result from Sonnet
 */
export interface SummaryResult {
  paper_id: number;
  summary: string;
  model: string;
  tokens_input: number;
  tokens_output: number;
  cost_usd: number;
}

// =============================================================================
// MCP Tool Request/Response Types
// =============================================================================

/**
 * Request for list_library MCP tool
 */
export interface ListLibraryRequest {
  limit?: number;
  offset?: number;
  explored?: boolean | null; // null = all, true = only explored, false = only unexplored
  bookmarked?: boolean | null;
}

/**
 * Request for search_papers MCP tool
 */
export interface SearchPapersRequest {
  query: string;
  limit?: number;
}

/**
 * Request for mark_explored MCP tool
 */
export interface MarkExploredRequest {
  paper_ids: number[];
  explored: boolean;
}

// =============================================================================
// Error Types
// =============================================================================

/**
 * Standardized API error response
 */
export interface ApiError {
  error: string;
  code: string;
  details?: Record<string, unknown>;
}

// =============================================================================
// MCP JSON-RPC Protocol Types
// =============================================================================

/**
 * JSON-RPC 2.0 request format
 */
export interface JsonRpcRequest {
  jsonrpc: '2.0';
  method: string;
  params?: Record<string, unknown>;
  id: string | number | null;
}

/**
 * JSON-RPC 2.0 success response
 */
export interface JsonRpcSuccessResponse {
  jsonrpc: '2.0';
  result: unknown;
  id: string | number | null;
}

/**
 * JSON-RPC 2.0 error response
 */
export interface JsonRpcErrorResponse {
  jsonrpc: '2.0';
  error: {
    code: number;
    message: string;
    data?: unknown;
  };
  id: string | number | null;
}

/**
 * JSON-RPC standard error codes
 */
export const JsonRpcErrorCodes = {
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,
} as const;

/**
 * MCP initialize response
 */
export interface McpInitializeResult {
  protocolVersion: string;
  capabilities: {
    tools?: Record<string, unknown>;
  };
  serverInfo: {
    name: string;
    version: string;
  };
}

/**
 * MCP tool definition with JSON Schema
 */
export interface McpToolDefinition {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
  };
}

/**
 * MCP tools/list response
 */
export interface McpToolsListResult {
  tools: McpToolDefinition[];
}

/**
 * MCP tools/call request params
 */
export interface McpToolsCallParams {
  name: string;
  arguments?: Record<string, unknown>;
}

/**
 * MCP content block (text type)
 */
export interface McpTextContent {
  type: 'text';
  text: string;
}

/**
 * MCP tools/call response
 */
export interface McpToolsCallResult {
  content: McpTextContent[];
  isError?: boolean;
}
