// =============================================================================
// kivv - MCP Server: JSON-RPC Protocol Handler
// =============================================================================
// Implements MCP JSON-RPC protocol for standard MCP clients (mcp-remote, etc.)
// Supports: initialize, tools/list, tools/call methods
// =============================================================================

import { Context } from 'hono';
import {
  Env,
  User,
  JsonRpcRequest,
  JsonRpcSuccessResponse,
  JsonRpcErrorResponse,
  JsonRpcErrorCodes,
  McpInitializeResult,
  McpToolsListResult,
  McpToolDefinition,
  McpToolsCallParams,
  McpToolsCallResult,
  ListLibraryRequest,
} from '../../shared/types';
import {
  DEFAULT_PAGE_LIMIT,
  MAX_PAGE_LIMIT,
} from '../../shared/constants';

// =============================================================================
// MCP Protocol Constants
// =============================================================================

const MCP_PROTOCOL_VERSION = '2024-11-05';
const SERVER_NAME = 'kivv-mcp';
const SERVER_VERSION = '1.0.0';

// =============================================================================
// Tool Definitions
// =============================================================================

const TOOL_DEFINITIONS: McpToolDefinition[] = [
  {
    name: 'list_library',
    description: 'Query user\'s paper library with optional filters and pagination. Returns papers from your arXiv research library.',
    inputSchema: {
      type: 'object',
      properties: {
        limit: {
          type: 'number',
          description: `Maximum number of papers to return (default: ${DEFAULT_PAGE_LIMIT}, max: ${MAX_PAGE_LIMIT})`,
        },
        offset: {
          type: 'number',
          description: 'Number of papers to skip for pagination (default: 0)',
        },
        explored: {
          type: 'boolean',
          description: 'Filter by exploration status. true = only explored, false = only unexplored, omit = all papers',
        },
        bookmarked: {
          type: 'boolean',
          description: 'Filter by bookmark status. true = only bookmarked, false = only unbookmarked, omit = all papers',
        },
      },
      required: [],
    },
  },
  {
    name: 'search_papers',
    description: 'Search papers in your library by keywords. Searches in title and abstract fields.',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search keywords to find in paper titles and abstracts',
        },
        limit: {
          type: 'number',
          description: `Maximum number of papers to return (default: ${DEFAULT_PAGE_LIMIT}, max: ${MAX_PAGE_LIMIT})`,
        },
        offset: {
          type: 'number',
          description: 'Number of papers to skip for pagination (default: 0)',
        },
        explored: {
          type: 'boolean',
          description: 'Filter by exploration status',
        },
        bookmarked: {
          type: 'boolean',
          description: 'Filter by bookmark status',
        },
      },
      required: ['query'],
    },
  },
  {
    name: 'mark_explored',
    description: 'Update a paper\'s status. Mark as explored/unexplored, bookmark/unbookmark, or add notes.',
    inputSchema: {
      type: 'object',
      properties: {
        paper_id: {
          type: 'number',
          description: 'The ID of the paper to update',
        },
        explored: {
          type: 'boolean',
          description: 'Mark paper as explored (true) or unexplored (false)',
        },
        bookmarked: {
          type: 'boolean',
          description: 'Mark paper as bookmarked (true) or unbookmarked (false)',
        },
        notes: {
          type: 'string',
          description: 'Add or update notes for this paper. Set to null to clear notes.',
        },
      },
      required: ['paper_id'],
    },
  },
];

// =============================================================================
// JSON-RPC Response Helpers
// =============================================================================

function createSuccessResponse(
  result: unknown,
  id: string | number | null
): JsonRpcSuccessResponse {
  return {
    jsonrpc: '2.0',
    result,
    id,
  };
}

function createErrorResponse(
  code: number,
  message: string,
  id: string | number | null,
  data?: unknown
): JsonRpcErrorResponse {
  return {
    jsonrpc: '2.0',
    error: { code, message, data },
    id,
  };
}

// =============================================================================
// MCP Method Handlers
// =============================================================================

function handleInitialize(): McpInitializeResult {
  return {
    protocolVersion: MCP_PROTOCOL_VERSION,
    capabilities: {
      tools: {},
    },
    serverInfo: {
      name: SERVER_NAME,
      version: SERVER_VERSION,
    },
  };
}

function handleToolsList(): McpToolsListResult {
  return {
    tools: TOOL_DEFINITIONS,
  };
}

async function handleToolsCall(
  params: McpToolsCallParams,
  user: User,
  env: Env
): Promise<McpToolsCallResult> {
  const { name, arguments: args = {} } = params;

  switch (name) {
    case 'list_library':
      return await executeListLibrary(args as ListLibraryRequest, user, env);
    case 'search_papers':
      return await executeSearchPapers(args as { query: string; limit?: number; offset?: number; explored?: boolean; bookmarked?: boolean }, user, env);
    case 'mark_explored':
      return await executeMarkExplored(args as { paper_id: number; explored?: boolean; bookmarked?: boolean; notes?: string | null }, user, env);
    default:
      return {
        content: [{ type: 'text', text: `Unknown tool: ${name}` }],
        isError: true,
      };
  }
}

// =============================================================================
// Tool Execution (Reuses existing logic)
// =============================================================================

async function executeListLibrary(
  args: ListLibraryRequest,
  user: User,
  env: Env
): Promise<McpToolsCallResult> {
  const limit = Math.min(args.limit ?? DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT);
  const offset = args.offset ?? 0;

  // Build WHERE clause with user isolation
  const filters: string[] = ['ups.user_id = ?'];
  const bindings: (number | boolean)[] = [user.id];

  if (args.explored !== undefined && args.explored !== null) {
    filters.push('ups.explored = ?');
    bindings.push(args.explored ? 1 : 0);
  }

  if (args.bookmarked !== undefined && args.bookmarked !== null) {
    filters.push('ups.bookmarked = ?');
    bindings.push(args.bookmarked ? 1 : 0);
  }

  const whereClause = filters.join(' AND ');

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
      p.summary,
      ups.explored,
      ups.bookmarked,
      ups.notes
    FROM papers p
    INNER JOIN user_paper_status ups ON p.id = ups.paper_id
    WHERE ${whereClause}
    ORDER BY p.published_date DESC
    LIMIT ? OFFSET ?
  `;

  try {
    const queryBindings = [...bindings, limit, offset];
    const result = await env.DB.prepare(query).bind(...queryBindings).all();

    const countQuery = `
      SELECT COUNT(*) as total
      FROM papers p
      INNER JOIN user_paper_status ups ON p.id = ups.paper_id
      WHERE ${whereClause}
    `;
    const countResult = await env.DB.prepare(countQuery).bind(...bindings).first() as { total: number } | null;
    const total = countResult?.total ?? 0;

    const papers = result.results.map((paper: any) => ({
      ...paper,
      explored: Boolean(paper.explored),
      bookmarked: Boolean(paper.bookmarked),
    }));

    const responseText = JSON.stringify({
      papers,
      total,
      limit,
      offset,
    }, null, 2);

    return {
      content: [{ type: 'text', text: responseText }],
    };
  } catch (error) {
    console.error('[jsonrpc:list_library] Database error:', error);
    return {
      content: [{ type: 'text', text: 'Failed to query library' }],
      isError: true,
    };
  }
}

async function executeSearchPapers(
  args: { query: string; limit?: number; offset?: number; explored?: boolean; bookmarked?: boolean },
  user: User,
  env: Env
): Promise<McpToolsCallResult> {
  if (!args.query || typeof args.query !== 'string' || args.query.trim() === '') {
    return {
      content: [{ type: 'text', text: 'Query parameter is required and must be a non-empty string' }],
      isError: true,
    };
  }

  const limit = Math.min(args.limit ?? DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT);
  const offset = args.offset ?? 0;
  const trimmedQuery = args.query.trim();

  // Build WHERE clause
  const filters: string[] = ['ups.user_id = ?'];
  const bindings: (number | string)[] = [user.id];

  // Add search condition
  const escapedQuery = trimmedQuery.toLowerCase().replace(/%/g, ' ').replace(/_/g, ' ');
  const searchPattern = `%${escapedQuery}%`;
  filters.push('(LOWER(p.title) LIKE ? OR LOWER(p.abstract) LIKE ?)');
  bindings.push(searchPattern, searchPattern);

  if (args.explored !== undefined && args.explored !== null) {
    filters.push('ups.explored = ?');
    bindings.push(args.explored ? 1 : 0);
  }

  if (args.bookmarked !== undefined && args.bookmarked !== null) {
    filters.push('ups.bookmarked = ?');
    bindings.push(args.bookmarked ? 1 : 0);
  }

  const whereClause = filters.join(' AND ');

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
      p.summary,
      ups.explored,
      ups.bookmarked,
      ups.notes
    FROM papers p
    INNER JOIN user_paper_status ups ON p.id = ups.paper_id
    WHERE ${whereClause}
    ORDER BY p.published_date DESC
    LIMIT ? OFFSET ?
  `;

  try {
    const queryBindings = [...bindings, limit, offset];
    const result = await env.DB.prepare(query).bind(...queryBindings).all();

    const countQuery = `
      SELECT COUNT(*) as total
      FROM papers p
      INNER JOIN user_paper_status ups ON p.id = ups.paper_id
      WHERE ${whereClause}
    `;
    const countResult = await env.DB.prepare(countQuery).bind(...bindings).first() as { total: number } | null;
    const total = countResult?.total ?? 0;

    const papers = result.results.map((paper: any) => ({
      ...paper,
      explored: Boolean(paper.explored),
      bookmarked: Boolean(paper.bookmarked),
    }));

    const responseText = JSON.stringify({
      papers,
      total,
      limit,
      offset,
      query: trimmedQuery,
    }, null, 2);

    return {
      content: [{ type: 'text', text: responseText }],
    };
  } catch (error) {
    console.error('[jsonrpc:search_papers] Database error:', error);
    return {
      content: [{ type: 'text', text: 'Failed to search papers' }],
      isError: true,
    };
  }
}

async function executeMarkExplored(
  args: { paper_id: number; explored?: boolean; bookmarked?: boolean; notes?: string | null },
  user: User,
  env: Env
): Promise<McpToolsCallResult> {
  if (args.paper_id === undefined || args.paper_id === null) {
    return {
      content: [{ type: 'text', text: 'paper_id is required' }],
      isError: true,
    };
  }

  if (typeof args.paper_id !== 'number' || !Number.isInteger(args.paper_id) || args.paper_id < 1) {
    return {
      content: [{ type: 'text', text: 'paper_id must be a positive integer' }],
      isError: true,
    };
  }

  try {
    // Verify paper exists
    const paper = await env.DB
      .prepare('SELECT id FROM papers WHERE id = ?')
      .bind(args.paper_id)
      .first();

    if (!paper) {
      return {
        content: [{ type: 'text', text: `Paper with id ${args.paper_id} not found` }],
        isError: true,
      };
    }

    const now = new Date().toISOString();

    // Check if user_paper_status record exists
    const existingStatus = await env.DB
      .prepare('SELECT * FROM user_paper_status WHERE user_id = ? AND paper_id = ?')
      .bind(user.id, args.paper_id)
      .first();

    if (existingStatus) {
      // UPDATE existing record
      const updates: string[] = ['read_at = ?'];
      const bindings: (string | number | null)[] = [now];

      if (args.explored !== undefined) {
        updates.push('explored = ?');
        bindings.push(args.explored ? 1 : 0);
      }

      if (args.bookmarked !== undefined) {
        updates.push('bookmarked = ?');
        bindings.push(args.bookmarked ? 1 : 0);
      }

      if (args.notes !== undefined) {
        updates.push('notes = ?');
        bindings.push(args.notes === null ? null : args.notes);
      }

      bindings.push(user.id, args.paper_id);

      await env.DB
        .prepare(`UPDATE user_paper_status SET ${updates.join(', ')} WHERE user_id = ? AND paper_id = ?`)
        .bind(...bindings)
        .run();
    } else {
      // INSERT new record
      await env.DB
        .prepare(`
          INSERT INTO user_paper_status
          (user_id, paper_id, explored, bookmarked, notes, read_at, created_at)
          VALUES (?, ?, ?, ?, ?, ?, ?)
        `)
        .bind(
          user.id,
          args.paper_id,
          args.explored !== undefined ? (args.explored ? 1 : 0) : 0,
          args.bookmarked !== undefined ? (args.bookmarked ? 1 : 0) : 0,
          args.notes !== undefined ? args.notes : null,
          now,
          now
        )
        .run();
    }

    // Fetch updated status
    const updatedStatus = await env.DB
      .prepare('SELECT * FROM user_paper_status WHERE user_id = ? AND paper_id = ?')
      .bind(user.id, args.paper_id)
      .first() as {
        explored: number;
        bookmarked: number;
        notes: string | null;
        read_at: string | null;
      } | null;

    const responseText = JSON.stringify({
      success: true,
      paper_id: args.paper_id,
      status: {
        explored: Boolean(updatedStatus?.explored),
        bookmarked: Boolean(updatedStatus?.bookmarked),
        notes: updatedStatus?.notes ?? null,
        read_at: updatedStatus?.read_at ?? null,
      },
    }, null, 2);

    return {
      content: [{ type: 'text', text: responseText }],
    };
  } catch (error) {
    console.error('[jsonrpc:mark_explored] Database error:', error);
    return {
      content: [{ type: 'text', text: 'Failed to update paper status' }],
      isError: true,
    };
  }
}

// =============================================================================
// Main JSON-RPC Handler
// =============================================================================

export async function handleJsonRpc(c: Context) {
  const user = c.get('user') as User;
  const env = c.env as Env;

  // Parse JSON-RPC request
  let request: JsonRpcRequest;
  try {
    request = await c.req.json<JsonRpcRequest>();
  } catch (error) {
    return c.json(
      createErrorResponse(JsonRpcErrorCodes.PARSE_ERROR, 'Parse error', null),
      200
    );
  }

  // Validate JSON-RPC request structure
  if (request.jsonrpc !== '2.0' || typeof request.method !== 'string') {
    return c.json(
      createErrorResponse(JsonRpcErrorCodes.INVALID_REQUEST, 'Invalid Request', request.id ?? null),
      200
    );
  }

  const { method, params, id } = request;

  // Handle MCP methods
  try {
    switch (method) {
      case 'initialize':
        return c.json(createSuccessResponse(handleInitialize(), id), 200);

      case 'tools/list':
        return c.json(createSuccessResponse(handleToolsList(), id), 200);

      case 'tools/call': {
        if (!params || typeof params.name !== 'string') {
          return c.json(
            createErrorResponse(JsonRpcErrorCodes.INVALID_PARAMS, 'Invalid params: name is required', id),
            200
          );
        }
        const result = await handleToolsCall(params as unknown as McpToolsCallParams, user, env);
        return c.json(createSuccessResponse(result, id), 200);
      }

      case 'notifications/initialized':
        // Client notification that initialization is complete - no response needed
        return c.json(createSuccessResponse({}, id), 200);

      default:
        return c.json(
          createErrorResponse(JsonRpcErrorCodes.METHOD_NOT_FOUND, `Method not found: ${method}`, id),
          200
        );
    }
  } catch (error) {
    console.error('[jsonrpc] Error handling method:', method, error);
    return c.json(
      createErrorResponse(JsonRpcErrorCodes.INTERNAL_ERROR, 'Internal error', id),
      200
    );
  }
}
