-- Run this entire script in Supabase → SQL Editor.
-- Creates the vector search RPC used by rag_engine.py / Streamlit chat.

create extension if not exists vector;

create or replace function public.match_fda_document_chunks_v1_naive(
  match_count int,
  query_embedding halfvec(3072)
)
returns table (
  id uuid,
  brand_name text,
  generic_name text,
  manufacturer_name text,
  section_name text,
  chunk_content text,
  metadata jsonb,
  similarity double precision
)
language sql
stable
as $$
  select
    c.id,
    c.brand_name,
    c.generic_name,
    c.manufacturer_name,
    c.section_name,
    c.chunk_content,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.fda_document_chunks c
  where c.metadata->>'pipeline_version' = 'v1.0_naive'
  order by c.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

grant execute on function public.match_fda_document_chunks_v1_naive(int, halfvec)
  to anon, authenticated, service_role;

notify pgrst, 'reload schema';
