-- Run in the Supabase SQL editor before populate_vector_db.py
create extension if not exists vector;

create table if not exists public.fda_document_chunks (
  id uuid primary key default gen_random_uuid(),
  brand_name text,
  generic_name text,
  manufacturer_name text,
  section_name text not null,
  chunk_content text not null,
  embedding halfvec(3072) not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists fda_document_chunks_embedding_idx
  on public.fda_document_chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

create index if not exists fda_document_chunks_brand_name_idx
  on public.fda_document_chunks (brand_name);

create index if not exists fda_document_chunks_section_name_idx
  on public.fda_document_chunks (section_name);
