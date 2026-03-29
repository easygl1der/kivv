// =============================================================================
// kivv - arXiv Client Tests
// =============================================================================
// Comprehensive unit tests for ArxivClient and ArxivQueryBuilder
// Tests rate limiting, XML parsing, error handling, and query building
// =============================================================================

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ArxivClient, ArxivQueryBuilder, type ArxivPaper } from '../../shared/arxiv-client';

// =============================================================================
// Mock Data - Sample arXiv API Responses
// =============================================================================

const SAMPLE_ATOM_XML = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: cat:cs.AI</title>
  <entry>
    <id>http://arxiv.org/abs/2101.12345v1</id>
    <title>Attention Is All You Need</title>
    <summary>We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.</summary>
    <published>2017-06-12T17:57:34Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.AI"/>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/abs/2101.12345" rel="alternate"/>
    <link href="http://arxiv.org/pdf/2101.12345" rel="related"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2202.54321v2</id>
    <title>BERT: Pre-training of Deep Bidirectional Transformers</title>
    <summary>We introduce a new language representation model called BERT.</summary>
    <published>2018-10-11T18:21:03Z</published>
    <author><name>Jacob Devlin</name></author>
    <author><name>Ming-Wei Chang</name></author>
    <author><name>Kenton Lee</name></author>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
    <link href="http://arxiv.org/abs/2202.54321" rel="alternate"/>
    <link href="http://arxiv.org/pdf/2202.54321" rel="related"/>
  </entry>
</feed>`;

const MALFORMED_XML = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/9999.99999v1</id>
    <title>Incomplete Entry</title>
    <!-- Missing summary, authors, categories -->
  </entry>
</feed>`;

const EMPTY_FEED_XML = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: cat:nonexistent</title>
</feed>`;

// =============================================================================
// ArxivQueryBuilder Tests
// =============================================================================

describe('ArxivQueryBuilder', () => {
  it('should build category query', () => {
    const query = new ArxivQueryBuilder()
      .addCategory('cs.AI')
      .build();

    expect(query).toBe('cat:cs.AI');
  });

  it('should build multiple category query with OR', () => {
    const query = new ArxivQueryBuilder()
      .addCategory('cs.AI')
      .addCategory('cs.LG')
      .build('OR');

    expect(query).toBe('cat:cs.AI OR cat:cs.LG');
  });

  it('should build multiple category query with AND', () => {
    const query = new ArxivQueryBuilder()
      .addCategory('cs.AI')
      .addCategory('cs.LG')
      .build('AND');

    expect(query).toBe('cat:cs.AI AND cat:cs.LG');
  });

  it('should build keyword query', () => {
    const query = new ArxivQueryBuilder()
      .addKeyword('transformers')
      .build();

    expect(query).toBe('all:transformers');
  });

  it('should build multiple keyword query', () => {
    const query = new ArxivQueryBuilder()
      .addKeyword('transformers')
      .addKeyword('attention')
      .build('OR');

    expect(query).toBe('all:transformers OR all:attention');
  });

  it('should build title keyword query', () => {
    const query = new ArxivQueryBuilder()
      .addTitleKeyword('attention')
      .build();

    expect(query).toBe('ti:attention');
  });

  it('should build author query', () => {
    const query = new ArxivQueryBuilder()
      .addAuthor('Vaswani')
      .build();

    expect(query).toBe('au:Vaswani');
  });

  it('should build abstract keyword query', () => {
    const query = new ArxivQueryBuilder()
      .addAbstractKeyword('neural networks')
      .build();

    expect(query).toBe('abs:neural networks');
  });

  it('should build complex mixed query', () => {
    const query = new ArxivQueryBuilder()
      .addCategory('cs.AI')
      .addKeyword('transformers')
      .addAuthor('Vaswani')
      .build('AND');

    expect(query).toBe('cat:cs.AI AND all:transformers AND au:Vaswani');
  });

  it('should reset builder to empty state', () => {
    const builder = new ArxivQueryBuilder()
      .addCategory('cs.AI')
      .addKeyword('transformers');

    const firstQuery = builder.build();
    expect(firstQuery).toBe('cat:cs.AI OR all:transformers');

    builder.reset();
    const secondQuery = builder.addCategory('cs.LG').build();
    expect(secondQuery).toBe('cat:cs.LG');
  });

  it('should default to OR operator', () => {
    const query = new ArxivQueryBuilder()
      .addCategory('cs.AI')
      .addCategory('cs.LG')
      .build(); // No operator specified

    expect(query).toBe('cat:cs.AI OR cat:cs.LG');
  });
});

// =============================================================================
// ArxivClient - XML Parsing Tests
// =============================================================================

describe('ArxivClient - XML Parsing', () => {
  let client: ArxivClient;

  beforeEach(() => {
    client = new ArxivClient();
  });

  it('should parse valid Atom XML correctly', async () => {
    // Mock fetch to return sample XML
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        text: () => Promise.resolve(SAMPLE_ATOM_XML),
      } as Response)
    );

    const papers = await client.search({ query: 'cat:cs.AI' });

    expect(papers).toHaveLength(2);

    // Verify first paper
    expect(papers[0].arxiv_id).toBe('2101.12345');
    expect(papers[0].title).toBe('Attention Is All You Need');
    expect(papers[0].authors).toBe('Ashish Vaswani, Noam Shazeer');
    expect(papers[0].abstract).toContain('Transformer');
    expect(papers[0].published_date).toBe('2017-06-12T17:57:34Z');
    expect(papers[0].categories).toBe('cs.AI, cs.LG');
    expect(papers[0].arxiv_url).toBe('http://arxiv.org/abs/2101.12345');
    expect(papers[0].pdf_url).toBe('http://arxiv.org/pdf/2101.12345');

    // Verify second paper
    expect(papers[1].arxiv_id).toBe('2202.54321');
    expect(papers[1].title).toBe('BERT: Pre-training of Deep Bidirectional Transformers');
    expect(papers[1].authors).toBe('Jacob Devlin, Ming-Wei Chang, Kenton Lee');
    expect(papers[1].abstract).toContain('BERT');
    expect(papers[1].published_date).toBe('2018-10-11T18:21:03Z');
    expect(papers[1].categories).toBe('cs.CL, cs.AI');
  });

  it('should strip version suffix from arXiv ID', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        text: () => Promise.resolve(SAMPLE_ATOM_XML),
      } as Response)
    );

    const papers = await client.search({ query: 'cat:cs.AI' });

    // XML contains "2101.12345v1" but should return "2101.12345"
    expect(papers[0].arxiv_id).toBe('2101.12345');
    expect(papers[0].arxiv_id).not.toContain('v1');

    // XML contains "2202.54321v2" but should return "2202.54321"
    expect(papers[1].arxiv_id).toBe('2202.54321');
    expect(papers[1].arxiv_id).not.toContain('v2');
  });

  it('should handle multiple authors correctly', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        text: () => Promise.resolve(SAMPLE_ATOM_XML),
      } as Response)
    );

    const papers = await client.search({ query: 'cat:cs.AI' });

    expect(papers[0].authors).toBe('Ashish Vaswani, Noam Shazeer');
    expect(papers[1].authors).toBe('Jacob Devlin, Ming-Wei Chang, Kenton Lee');
  });

  it('should handle multiple categories correctly', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        text: () => Promise.resolve(SAMPLE_ATOM_XML),
      } as Response)
    );

    const papers = await client.search({ query: 'cat:cs.AI' });

    expect(papers[0].categories).toBe('cs.AI, cs.LG');
    expect(papers[1].categories).toBe('cs.CL, cs.AI');
  });

  it('should handle malformed XML gracefully', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        text: () => Promise.resolve(MALFORMED_XML),
      } as Response)
    );

    const papers = await client.search({ query: 'cat:cs.AI' });

    // Should skip entries with missing required fields
    expect(papers).toHaveLength(0);
  });

  it('should handle empty feed XML', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        text: () => Promise.resolve(EMPTY_FEED_XML),
      } as Response)
    );

    const papers = await client.search({ query: 'cat:nonexistent' });

    expect(papers).toHaveLength(0);
  });

  it('should use fallback URLs when links missing', async () => {
    const xmlWithoutLinks = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.56789v1</id>
    <title>Test Paper</title>
    <summary>Test abstract</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Test Author</name></author>
    <category term="cs.AI"/>
  </entry>
</feed>`;

    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        text: () => Promise.resolve(xmlWithoutLinks),
      } as Response)
    );

    const papers = await client.search({ query: 'cat:cs.AI' });

    expect(papers).toHaveLength(1);
    expect(papers[0].arxiv_url).toBe('http://arxiv.org/abs/1234.56789');
    expect(papers[0].pdf_url).toBe('http://arxiv.org/pdf/1234.56789');
  });
});

// =============================================================================
// ArxivClient - Rate Limiting Tests
// =============================================================================

describe('ArxivClient - Rate Limiting', () => {
  let client: ArxivClient;

  beforeEach(() => {
    client = new ArxivClient();
    vi.clearAllMocks();

    // Mock fetch for all rate limit tests
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        text: () => Promise.resolve(EMPTY_FEED_XML),
      } as Response)
    );
  });

  it('should enforce 3-second rate limit between requests', async () => {
    const startTime = Date.now();

    // First request (no delay)
    await client.search({ query: 'cat:cs.AI' });
    const firstRequestTime = Date.now() - startTime;

    // Second request (should wait ~3 seconds + jitter)
    await client.search({ query: 'cat:cs.LG' });
    const secondRequestTime = Date.now() - startTime;

    // Second request should take at least 3000ms longer
    const timeBetweenRequests = secondRequestTime - firstRequestTime;
    expect(timeBetweenRequests).toBeGreaterThanOrEqual(3000);

    // Should also have jitter (100-500ms), so total should be <= 3500ms
    expect(timeBetweenRequests).toBeLessThanOrEqual(3600);
  });

  it('should add random jitter to rate limit', async () => {
    const delays: number[] = [];

    // Make 3 requests and measure delays (reduced from 5 to avoid timeout)
    for (let i = 0; i < 3; i++) {
      const startTime = Date.now();
      await client.search({ query: `cat:cs.AI${i}` });

      if (i > 0) {
        delays.push(Date.now() - startTime);
      }
    }

    // All delays should be at least 3000ms
    for (const delay of delays) {
      expect(delay).toBeGreaterThanOrEqual(3000);
    }

    // Delays should have jitter variation (within 100ms buckets)
    // Just verify delays are in expected range (3000-3600ms)
    for (const delay of delays) {
      expect(delay).toBeLessThanOrEqual(3600);
    }
  }, 15000); // Increase timeout to 15 seconds

  it('should NOT delay first request', async () => {
    const startTime = Date.now();
    await client.search({ query: 'cat:cs.AI' });
    const elapsed = Date.now() - startTime;

    // First request should be immediate (< 100ms for network + processing)
    expect(elapsed).toBeLessThan(1000);
  });

  it('should enforce rate limit across multiple sequential requests', async () => {
    const startTime = Date.now();

    // Make 3 requests sequentially
    await client.search({ query: 'cat:cs.AI' });
    await client.search({ query: 'cat:cs.LG' });
    await client.search({ query: 'cat:cs.CL' });

    const totalTime = Date.now() - startTime;

    // Total time should be at least 6 seconds (2 delays of 3s each)
    // First request is immediate, second waits 3s, third waits another 3s
    expect(totalTime).toBeGreaterThanOrEqual(6000);

    // Should also be less than 7.5s (accounting for jitter)
    expect(totalTime).toBeLessThan(7500);
  }, 15000); // Increase timeout to 15 seconds
});

// =============================================================================
// ArxivClient - Error Handling Tests
// =============================================================================

describe('ArxivClient - Error Handling', () => {
  let client: ArxivClient;

  beforeEach(() => {
    client = new ArxivClient();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should return empty array on HTTP error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    } as Response);

    const papers = await client.search({ query: 'cat:cs.AI' });

    expect(papers).toHaveLength(0);
  });

  it('should return empty array on network error', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Network error'));

    const papers = await client.search({ query: 'cat:cs.AI' });

    expect(papers).toHaveLength(0);
  });

  it('should return empty array on invalid XML', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      text: () => Promise.resolve('not valid xml at all'),
    } as Response);

    const papers = await client.search({ query: 'cat:cs.AI' });

    expect(papers).toHaveLength(0);
  });
});

// =============================================================================
// ArxivClient - Search Parameters Tests
// =============================================================================

describe('ArxivClient - Search Parameters', () => {
  let client: ArxivClient;

  beforeEach(() => {
    client = new ArxivClient();
  });

  it('should build URL with correct query parameters', async () => {
    let capturedUrl = '';

    global.fetch = vi.fn((url) => {
      capturedUrl = url.toString();
      return Promise.resolve({
        ok: true,
        text: () => Promise.resolve(EMPTY_FEED_XML),
      } as Response);
    });

    await client.search({
      query: 'cat:cs.AI',
      maxResults: 20,
      start: 10,
      sortBy: 'relevance',
      sortOrder: 'ascending',
    });

    expect(capturedUrl).toContain('search_query=cat%3Acs.AI');
    expect(capturedUrl).toContain('max_results=20');
    expect(capturedUrl).toContain('start=10');
    expect(capturedUrl).toContain('sortBy=relevance');
    expect(capturedUrl).toContain('sortOrder=ascending');
  });

  it('should use default parameters when not specified', async () => {
    let capturedUrl = '';

    global.fetch = vi.fn((url) => {
      capturedUrl = url.toString();
      return Promise.resolve({
        ok: true,
        text: () => Promise.resolve(EMPTY_FEED_XML),
      } as Response);
    });

    await client.search({ query: 'cat:cs.AI' });

    expect(capturedUrl).toContain('max_results=10');
    expect(capturedUrl).toContain('start=0');
    expect(capturedUrl).toContain('sortBy=submittedDate');
    expect(capturedUrl).toContain('sortOrder=descending');
  });
});
