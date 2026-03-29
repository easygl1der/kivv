// =============================================================================
// kivv - MCP Server Authentication Middleware
// =============================================================================
// SECURITY-CRITICAL CODE - 100% test coverage required
// Implements API key authentication with:
// - Parameterized SQL queries (SQL injection prevention)
// - Active user check (is_active = 1)
// - Request logging for audit trail
// - No sensitive data leakage in error responses
// =============================================================================

import type { Env, User } from '../../shared/types';
import { createErrorResponse } from '../../shared/utils';
import {
  ERROR_INVALID_API_KEY,
  ERROR_USER_INACTIVE,
  ERROR_MISSING_AUTH,
  HTTP_UNAUTHORIZED,
  HTTP_FORBIDDEN,
} from '../../shared/constants';

// =============================================================================
// Core Authentication Functions
// =============================================================================

/**
 * Authenticate user by API key from request header
 * Returns User object if valid, null otherwise
 *
 * SECURITY FEATURES:
 * - Parameterized queries (SQL injection prevention)
 * - Active user check (is_active = 1)
 * - last_login timestamp update
 * - Error logging without sensitive data exposure
 *
 * @param request - Incoming HTTP request
 * @param env - Cloudflare Workers environment bindings
 * @returns User object if authenticated, null otherwise
 *
 * @example
 * const user = await authenticateUser(request, env);
 * if (!user) {
 *   return createUnauthorizedResponse('invalid');
 * }
 */
export async function authenticateUser(
  request: Request,
  env: Env
): Promise<User | null> {
  // Extract API key from x-api-key header
  const apiKey = request.headers.get('x-api-key');

  if (!apiKey) {
    return null;
  }

  // Empty string or whitespace-only API key should be rejected
  // Note: We don't trim the API key - whitespace is significant
  if (apiKey === '' || apiKey.trim() === '') {
    return null;
  }

  // Query database for user with this API key
  // SECURITY: Using parameterized query with .bind() to prevent SQL injection
  try {
    const result = await env.DB
      .prepare('SELECT * FROM users WHERE api_key = ? LIMIT 1')
      .bind(apiKey)
      .first() as User | null;

    if (!result) {
      return null;
    }

    // Check if user is active
    // SECURITY: Inactive users must be rejected (even with valid API key)
    if (!result.is_active) {
      return null;
    }

    // Update last_login timestamp
    // Note: In production this could be fire-and-forget for performance,
    // but for testing we need to await it to prevent storage isolation issues
    try {
      await env.DB
        .prepare('UPDATE users SET last_login = ? WHERE id = ?')
        .bind(new Date().toISOString(), result.id)
        .run();
    } catch (err) {
      console.error('Failed to update last_login:', err);
    }

    return result;
  } catch (error) {
    console.error('Database error during authentication:', error);
    return null;
  }
}

/**
 * Create 401 Unauthorized response for missing/invalid API key
 *
 * SECURITY: Generic error messages that don't leak user existence
 *
 * @param reason - Why authentication failed ('missing' | 'invalid')
 * @returns 401 Unauthorized Response
 */
export function createUnauthorizedResponse(
  reason: 'missing' | 'invalid'
): Response {
  const errorCode =
    reason === 'missing' ? ERROR_MISSING_AUTH : ERROR_INVALID_API_KEY;
  const message =
    reason === 'missing'
      ? 'API key required. Provide x-api-key header.'
      : 'Invalid or expired API key.';

  return createErrorResponse(message, errorCode, HTTP_UNAUTHORIZED);
}

/**
 * Create 403 Forbidden response for inactive user
 *
 * SECURITY: Generic message that doesn't leak user details
 *
 * @returns 403 Forbidden Response
 */
export function createForbiddenResponse(): Response {
  return createErrorResponse(
    'User account is inactive. Contact administrator.',
    ERROR_USER_INACTIVE,
    HTTP_FORBIDDEN
  );
}

// =============================================================================
// Hono Middleware Integration
// =============================================================================

/**
 * Hono middleware factory for authentication
 *
 * This middleware:
 * 1. Extracts API key from x-api-key header
 * 2. Validates against database
 * 3. Checks user is active
 * 4. Stores user in context for route handlers
 * 5. Logs successful authentication
 *
 * Usage: app.use('/mcp/*', createAuthMiddleware())
 *
 * @returns Hono middleware function
 *
 * @example
 * import { Hono } from 'hono';
 * import { createAuthMiddleware } from './auth';
 *
 * const app = new Hono();
 * app.use('/mcp/*', createAuthMiddleware());
 *
 * app.post('/mcp/tools/list_library', (c) => {
 *   const user = c.get('user'); // Authenticated user
 *   // ...
 * });
 */
export function createAuthMiddleware() {
  return async (c: any, next: any) => {
    const apiKey = c.req.header('x-api-key');

    if (!apiKey) {
      return createUnauthorizedResponse('missing');
    }

    const user = await authenticateUser(c.req.raw, c.env);

    if (!user) {
      return createUnauthorizedResponse('invalid');
    }

    if (!user.is_active) {
      return createForbiddenResponse();
    }

    // Store user in context for use in route handlers
    c.set('user', user);

    // Log authenticated request for audit trail
    // SECURITY: Log username and ID but NOT API key or sensitive data
    console.log(`[AUTH] User ${user.username} (ID: ${user.id}) authenticated`);

    await next();
  };
}
