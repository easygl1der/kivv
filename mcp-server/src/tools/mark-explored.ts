// =============================================================================
// kivv - MCP Server: mark_explored Tool
// =============================================================================
// Update paper exploration status (explored, bookmarked, notes)
// SECURITY-CRITICAL: User can only update their own paper status
// UPSERT Pattern: INSERT new status record or UPDATE existing record
// =============================================================================

import { Context } from 'hono';
import { Env, User } from '../../../shared/types';
import {
  HTTP_OK,
  HTTP_BAD_REQUEST,
  HTTP_NOT_FOUND,
  ERROR_INVALID_INPUT,
} from '../../../shared/constants';
import { createErrorResponse } from '../../../shared/utils';

// =============================================================================
// Request/Response Types
// =============================================================================

interface MarkExploredRequest {
  paper_id: number;          // REQUIRED - which paper to update
  explored?: boolean;        // Optional - mark as explored/unexplored
  bookmarked?: boolean;      // Optional - mark as bookmarked/unbookmarked
  notes?: string | null;     // Optional - add/update/clear notes
}

interface MarkExploredResponse {
  success: boolean;
  paper_id: number;
  status: {
    explored: boolean;
    bookmarked: boolean;
    notes: string | null;
    read_at: string | null;
  };
}

// =============================================================================
// Core Implementation
// =============================================================================

/**
 * mark_explored MCP Tool - Update paper exploration status
 *
 * Features:
 * - UPSERT pattern: creates new user_paper_status record or updates existing
 * - Update explored, bookmarked, notes fields independently
 * - Updates read_at timestamp on every modification
 * - User data isolation (user can only modify their own status)
 * - Paper existence validation
 * - SQL injection prevention (parameterized queries)
 *
 * @param c - Hono context with authenticated user
 * @returns JSON response with updated status
 *
 * @example
 * POST /mcp/tools/mark_explored
 * Headers: { "x-api-key": "user_api_key" }
 * Body: { "paper_id": 42, "explored": true, "notes": "Interesting approach" }
 *
 * Response:
 * {
 *   "success": true,
 *   "paper_id": 42,
 *   "status": {
 *     "explored": true,
 *     "bookmarked": false,
 *     "notes": "Interesting approach",
 *     "read_at": "2025-11-30T12:34:56.789Z"
 *   }
 * }
 */
export async function markExplored(c: Context) {
  // Get authenticated user from middleware
  const user = c.get('user') as User;

  // Parse request body
  let body: MarkExploredRequest;
  try {
    body = await c.req.json<MarkExploredRequest>();
  } catch (error) {
    return createErrorResponse(
      'Invalid JSON body',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  // Validate paper_id (REQUIRED)
  if (body.paper_id === undefined || body.paper_id === null) {
    return createErrorResponse(
      'paper_id is required',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  if (typeof body.paper_id !== 'number' || !Number.isInteger(body.paper_id) || body.paper_id < 1) {
    return createErrorResponse(
      'paper_id must be a positive integer',
      ERROR_INVALID_INPUT,
      HTTP_BAD_REQUEST
    );
  }

  try {
    // Verify paper exists
    const paper = await c.env.DB
      .prepare('SELECT id FROM papers WHERE id = ?')
      .bind(body.paper_id)
      .first();

    if (!paper) {
      return createErrorResponse(
        `Paper with id ${body.paper_id} not found`,
        'PAPER_NOT_FOUND',
        HTTP_NOT_FOUND
      );
    }

    const now = new Date().toISOString();

    // Check if user_paper_status record exists
    const existingStatus = await c.env.DB
      .prepare('SELECT * FROM user_paper_status WHERE user_id = ? AND paper_id = ?')
      .bind(user.id, body.paper_id)
      .first() as {
        user_id: number;
        paper_id: number;
        explored: number;
        bookmarked: number;
        notes: string | null;
        read_at: string | null;
        created_at: string;
      } | null;

    if (existingStatus) {
      // UPDATE existing record
      // Build dynamic SET clause based on provided fields
      const updates: string[] = ['read_at = ?'];
      const bindings: (string | number | null)[] = [now];

      if (body.explored !== undefined) {
        updates.push('explored = ?');
        bindings.push(body.explored ? 1 : 0);
      }

      if (body.bookmarked !== undefined) {
        updates.push('bookmarked = ?');
        bindings.push(body.bookmarked ? 1 : 0);
      }

      if (body.notes !== undefined) {
        updates.push('notes = ?');
        bindings.push(body.notes === null ? null : body.notes);
      }

      // Add WHERE clause bindings
      bindings.push(user.id, body.paper_id);

      await c.env.DB
        .prepare(`UPDATE user_paper_status SET ${updates.join(', ')} WHERE user_id = ? AND paper_id = ?`)
        .bind(...bindings)
        .run();
    } else {
      // INSERT new record
      await c.env.DB
        .prepare(`
          INSERT INTO user_paper_status
          (user_id, paper_id, explored, bookmarked, notes, read_at, created_at)
          VALUES (?, ?, ?, ?, ?, ?, ?)
        `)
        .bind(
          user.id,
          body.paper_id,
          body.explored !== undefined ? (body.explored ? 1 : 0) : 0,
          body.bookmarked !== undefined ? (body.bookmarked ? 1 : 0) : 0,
          body.notes !== undefined ? body.notes : null,
          now,
          now
        )
        .run();
    }

    // Fetch updated status
    const updatedStatus = await c.env.DB
      .prepare('SELECT * FROM user_paper_status WHERE user_id = ? AND paper_id = ?')
      .bind(user.id, body.paper_id)
      .first() as {
        user_id: number;
        paper_id: number;
        explored: number;
        bookmarked: number;
        notes: string | null;
        read_at: string | null;
        created_at: string;
      } | null;

    // This should never happen since we just created/updated it
    if (!updatedStatus) {
      return createErrorResponse(
        'Failed to retrieve updated status',
        'DATABASE_ERROR',
        500
      );
    }

    // Convert SQLite boolean integers to TypeScript booleans
    const response: MarkExploredResponse = {
      success: true,
      paper_id: body.paper_id,
      status: {
        explored: Boolean(updatedStatus.explored),
        bookmarked: Boolean(updatedStatus.bookmarked),
        notes: updatedStatus.notes,
        read_at: updatedStatus.read_at,
      },
    };

    return c.json(response, HTTP_OK);

  } catch (error) {
    console.error('[mark_explored] Database error:', error);
    return createErrorResponse(
      'Failed to update paper status',
      'DATABASE_ERROR',
      500
    );
  }
}
