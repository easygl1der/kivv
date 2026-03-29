-- Update topics for new research areas
-- Disable all current topics
UPDATE topics SET enabled = 0;

-- Insert new topics for mathematical research
-- User 1 (Jeff) gets the new topics
INSERT OR REPLACE INTO topics (user_id, topic_name, arxiv_query, enabled, relevance_threshold, max_papers_per_day) VALUES
  (1, 'Information Geometry', 'cat:stat.ML OR cat:math.ST OR cat:cs.IT AND (information geometry OR duistermaat-hanson OR wave facet OR spectral geometry)', 1, 0.5, 20),
  (1, 'Schubert Calculus', 'cat:math.AG AND (schubert OR degeneracy locus OR characteristic classes OR enumerative geometry OR giambelli)', 1, 0.5, 20),
  (1, 'Volatility Forecasting', 'cat:q-fin.MF OR cat:q-fin.ST OR cat:math.OC AND (volatility OR stochastic OR implied volatility OR garch OR options OR hedging)', 1, 0.5, 20),
  (1, 'High-Dimensional Statistics', 'cat:stat.ML OR cat:math.ST OR cat:cs.LG AND (high-dimensional OR high dimension OR sparsity OR compressed sensing OR random matrix OR curse of dimensionality)', 1, 0.5, 20);
