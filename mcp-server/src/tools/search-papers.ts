// =============================================================================
// kivv - MCP Server: search_papers Tool
// =============================================================================
// Search user's paper library by keywords in title/abstract
// SECURITY-CRITICAL: All queries filtered by authenticated user ID
// SQL INJECTION PREVENTION: Parameterized queries with LIKE wildcards
// =============================================================================

import { Context } from 'hono';
import { Env, User, PaperWithStatus } from '../../../shared/types';
import {
  HTTP_OK,
  HTTP_BAD_REQUEST,
  DEFAULT_PAGE_LIMIT,
  MAX_PAGE_LIMIT,
  ERROR_INVALID_INPUT,
} from '../../../shared/constants';
import { createErrorResponse } from '../../../shared/utils';

// =============================================================================
// Request/Response Types
// =============================================================================

interface SearchPapersRequest {
  query: string;           // REQUIRED - search keywords
  limit?: number;          // Optional, default 20, max 100
  offset?: number;         // Optional, default 0
  explored?: boolean;      // Optional filter
  bookmarked?: boolean;    // Optional filter
}

interface SearchPapersResponse {
  papers: PaperWithStatus[];
  total: number;
  limit: number;
  offset: number;
  query: string;
}

// =============================================================================
// Core Implementation
// =============================================================================

/**
 * search_papers MCP Tool - Search user's paper library by keywords
 *
 * Features:
 * - Case-insensitive keyword search in title and abstract
 * - User-isolated queries (JOIN with user_paper_status)
 * - Pagination (limit/offset)
 * - Filters (explored, bookmarked)
 * - SQL injection prevention (parameterized queries)
 * - Wildcard escaping for special characters
 *
 * @param c - Hono context with authenticated user
 * @returns JSON response with matching papers and pagination metadata
 *
 * @example
 * POST /mcp/tools/search_papers
 * Headers: { "x-api-key": "user_api_key" }
 * Body: { "query": "neural networks", "limit": 20, "explored": true }
 *
 * Response:
 * {
 *   "papers": [...],
 *   "total": 42,
 *   "limit": 20,
 *   "offset": 0,
 *   "query": "neural networks"
 * }
 */
export async function searchPapers(c: Context) {
  // Get authenticated user from middleware
  const user = c.get('user') as User;

  // Parse request body
  let body: SearchPapersRequest;
  try {
    body = await c.req.json<SearchPapersRequest>();
  } catch (error) {
    return createErrorResponse(
      'Invalid JSON body',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  // Validate query parameter (REQUIRED)
  if (body.query === undefined || body.query === null) {
    return createErrorResponse(
      'Query parameter is required',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  if (typeof body.query !== 'string') {
    return createErrorResponse(
      'Query parameter must be a string',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  const trimmedQuery = body.query.trim();
  if (trimmedQuery === '') {
    return createErrorResponse(
      'Query parameter cannot be empty',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  // Prevent overly long queries that could cause SQLite LIKE pattern complexity errors
  if (trimmedQuery.length > 500) {
    return createErrorResponse(
      'Query parameter is too long (max 500 characters)',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  // Validate and sanitize pagination parameters
  const limit = validateLimit(body.limit);
  const offset = validateOffset(body.offset);

  if (limit === null || offset === null) {
    return createErrorResponse(
      'Invalid pagination parameters. limit must be 1-100, offset must be >= 0.',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  // Build WHERE clause with user isolation
  const filters: string[] = ['ups.user_id = ?'];
  const bindings: (number | string)[] = [user.id];

  // Add search condition (case-insensitive LIKE search)
  // SQL INJECTION PREVENTION: Use parameterized query with wildcards
  // Escape SQL wildcards (%, _) in user input to prevent unintended pattern matching
  const escapedQuery = escapeSqlWildcards(trimmedQuery.toLowerCase());
  const searchPattern = `%${escapedQuery}%`;

  filters.push('(LOWER(p.title) LIKE ? OR LOWER(p.abstract) LIKE ?)');
  bindings.push(searchPattern, searchPattern);

  // Add optional explored filter
  if (body.explored !== undefined && body.explored !== null) {
    filters.push('ups.explored = ?');
    bindings.push(body.explored ? 1 : 0);
  }

  // Add optional bookmarked filter
  if (body.bookmarked !== undefined && body.bookmarked !== null) {
    filters.push('ups.bookmarked = ?');
    bindings.push(body.bookmarked ? 1 : 0);
  }

  const whereClause = filters.join(' AND ');

  // Execute main query with JOIN
  // SECURITY: All queries filter by user_id to prevent cross-user access
  const query = `
    SELECT
      p.id,
      p.arxiv_id,
      p.title,
      p.authors,
      p.abstract,
      p.categories,
      p.published_date,
      p.pdf_url,
      p.r2_key,
      p.summary,
      p.summary_generated_at,
      p.summary_model,
      p.relevance_score,
      p.content_hash,
      p.collected_for_user_id,
      p.created_at,
      ups.explored,
      ups.bookmarked,
      ups.notes,
      ups.read_at
    FROM papers p
    INNER JOIN user_paper_status ups ON p.id = ups.paper_id
    WHERE ${whereClause}
    ORDER BY p.published_date DESC
    LIMIT ? OFFSET ?
  `;

  try {
    // Add pagination parameters to bindings
    const queryBindings = [...bindings, limit, offset];

    const result = await (c.env.DB.prepare(query) as any).bind(...queryBindings).all() as { results: PaperWithStatus[] };

    // Get total count for pagination metadata
    const countQuery = `
      SELECT COUNT(*) as total
      FROM papers p
      INNER JOIN user_paper_status ups ON p.id = ups.paper_id
      WHERE ${whereClause}
    `;

    // Count query uses all bindings except limit/offset
    const countBindings = bindings;

    const countResult = await c.env.DB
      .prepare(countQuery)
      .bind(...countBindings)
      .first() as { total: number } | null;

    const total = countResult?.total ?? 0;

    // Convert SQLite boolean integers (0/1) to TypeScript booleans
    const papers = result.results.map(paper => ({
      ...paper,
      explored: Boolean(paper.explored),
      bookmarked: Boolean(paper.bookmarked),
    }));

    const response: SearchPapersResponse = {
      papers,
      total,
      limit,
      offset,
      query: trimmedQuery,
    };

    return c.json(response, HTTP_OK);

  } catch (error) {
    console.error('[search_papers] Database error:', error);
    return createErrorResponse(
      'Failed to search papers',
      'DATABASE_ERROR',
      500
    );
  }
}

// =============================================================================
// Validation Helpers
// =============================================================================

/**
 * Validate limit parameter
 * @param limit - Requested limit (undefined = default)
 * @returns Validated limit or null if invalid
 */
function validateLimit(limit: number | undefined): number | null {
  if (limit === undefined) {
    return DEFAULT_PAGE_LIMIT;
  }

  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_PAGE_LIMIT) {
    return null;
  }

  return limit;
}

/**
 * Validate offset parameter
 * @param offset - Requested offset (undefined = 0)
 * @returns Validated offset or null if invalid
 */
function validateOffset(offset: number | undefined): number | null {
  if (offset === undefined) {
    return 0;
  }

  if (!Number.isInteger(offset) || offset < 0) {
    return null;
  }

  return offset;
}

/**
 * Escape SQL LIKE wildcards in user input
 * Prevents unintended pattern matching when user searches for literal % or _
 *
 * Note: SQLite requires literal string escaping in the pattern itself.
 * We don't use ESCAPE clause, so we just remove wildcards to make them literal.
 *
 * @param input - User search query
 * @returns Input with wildcards removed for literal matching
 *
 * @example
 * escapeSqlWildcards('50% complete') // => '50 complete'
 * escapeSqlWildcards('test_file')    // => 'test file'
 */
function escapeSqlWildcards(input: string): string {
  // SQLite LIKE patterns: % and _ are wildcards
  // Since we can't easily escape them in parameterized queries,
  // we replace them with spaces to allow matching the surrounding text
  return input
    .replace(/%/g, ' ')    // Replace % with space for literal matching
    .replace(/_/g, ' ');   // Replace _ with space for literal matching
}
