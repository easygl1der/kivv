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
    authors: string;
    abstract: string;
    categories: string;
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
    date: string;
    user_id: number | null;
    service: string;
    papers_processed: number;
    tokens_input: number;
    tokens_output: number;
    cost_usd: number;
    created_at: string;
}
/**
 * Environment bindings for Cloudflare Workers
 * Includes D1 database, KV namespace, R2 bucket, and environment variables
 */
export interface Env {
    DB: D1Database;
    CACHE: KVNamespace;
    PAPERS: R2Bucket;
    MINIMAX_API_KEY: string;
    D1_DATABASE_ID: string;
    KV_NAMESPACE_ID: string;
    R2_BUCKET_NAME: string;
}
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
/**
 * Request for list_library MCP tool
 */
export interface ListLibraryRequest {
    limit?: number;
    offset?: number;
    explored?: boolean | null;
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
/**
 * Standardized API error response
 */
export interface ApiError {
    error: string;
    code: string;
    details?: Record<string, unknown>;
}
