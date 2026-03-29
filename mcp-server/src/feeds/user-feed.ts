// =============================================================================
// kivv - User Feed Endpoint Handlers
// =============================================================================
// Handles RSS and Atom feed generation for user paper libraries
// Public endpoints (no authentication required)
// =============================================================================

import { Context } from 'hono';
import { PaperWithStatus } from '../../../shared/types';
import { generateRSS, generateAtom } from './rss-generator';
import { HTTP_BAD_REQUEST, HTTP_NOT_FOUND, HTTP_OK } from '../../../shared/constants';

// =============================================================================
// Username Validation
// =============================================================================

/**
 * Validate username to prevent SQL injection and path traversal attacks
 * Only allows alphanumeric characters and underscores
 *
 * @param username - Username to validate
 * @returns true if username is valid, false otherwise
 */
function isValidUsername(username: string): boolean {
  if (!username || username.length === 0) return false;
  if (username.length > 50) return false; // Reasonable max length
  return /^[a-zA-Z0-9_]+$/.test(username);
}

// =============================================================================
// RSS Feed Endpoint
// =============================================================================

/**
 * GET /feeds/:username/rss.xml
 *
 * Generate RSS 2.0 feed for a user's paper library
 * Public endpoint - no authentication required
 *
 * @param c - Hono context
 * @returns RSS 2.0 XML response
 */
export async function getUserRSSFeed(c: Context) {
  const username = c.req.param('username');

  // Validate username to prevent injection attacks
  if (!isValidUsername(username)) {
    return c.text('Invalid username format. Only alphanumeric characters and underscores allowed.', HTTP_BAD_REQUEST);
  }

  try {
    // Find user by username
    const user = await c.env.DB
      .prepare('SELECT id FROM users WHERE username = ?')
      .bind(username)
      .first() as { id: number } | null;

    if (!user) {
      return c.text('User not found', HTTP_NOT_FOUND);
    }

    // Fetch user's papers (latest 50, newest first)
    const result = await c.env.DB
      .prepare(`
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
          ups.notes
        FROM papers p
        LEFT JOIN user_paper_status ups ON p.id = ups.paper_id AND ups.user_id = ?
        WHERE p.collected_for_user_id = ?
        ORDER BY p.published_date DESC
        LIMIT 50
      `)
      .bind(user.id, user.id)
      .all() as { results: PaperWithStatus[] };

    const papers = result.results || [];

    // Generate RSS XML
    const rssXml = generateRSS(username, papers);

    // Return RSS feed with correct MIME type
    return c.text(rssXml, HTTP_OK, {
      'Content-Type': 'application/rss+xml; charset=utf-8',
      'Cache-Control': 'public, max-age=300', // Cache for 5 minutes
    });
  } catch (error) {
    console.error('Error generating RSS feed:', error);
    return c.text('Internal server error', 500);
  }
}

// =============================================================================
// Atom Feed Endpoint
// =============================================================================

/**
 * GET /feeds/:username/atom.xml
 *
 * Generate Atom 1.0 feed for a user's paper library
 * Public endpoint - no authentication required
 *
 * @param c - Hono context
 * @returns Atom 1.0 XML response
 */
export async function getUserAtomFeed(c: Context) {
  const username = c.req.param('username');

  // Validate username to prevent injection attacks
  if (!isValidUsername(username)) {
    return c.text('Invalid username format. Only alphanumeric characters and underscores allowed.', HTTP_BAD_REQUEST);
  }

  try {
    // Find user by username
    const user = await c.env.DB
      .prepare('SELECT id FROM users WHERE username = ?')
      .bind(username)
      .first() as { id: number } | null;

    if (!user) {
      return c.text('User not found', HTTP_NOT_FOUND);
    }

    // Fetch user's papers (latest 50, newest first)
    const result = await c.env.DB
      .prepare(`
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
          ups.notes
        FROM papers p
        LEFT JOIN user_paper_status ups ON p.id = ups.paper_id AND ups.user_id = ?
        WHERE p.collected_for_user_id = ?
        ORDER BY p.published_date DESC
        LIMIT 50
      `)
      .bind(user.id, user.id)
      .all() as { results: PaperWithStatus[] };

    const papers = result.results || [];

    // Generate Atom XML
    const atomXml = generateAtom(username, papers);

    // Return Atom feed with correct MIME type
    return c.text(atomXml, HTTP_OK, {
      'Content-Type': 'application/atom+xml; charset=utf-8',
      'Cache-Control': 'public, max-age=300', // Cache for 5 minutes
    });
  } catch (error) {
    console.error('Error generating Atom feed:', error);
    return c.text('Internal server error', 500);
  }
}
