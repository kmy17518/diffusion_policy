"""Frozen CLIP task embeddings and checkpoint-local prompt provenance."""

import json
from pathlib import Path
import re

import torch


CLIP_MODEL = 'openai/clip-vit-large-patch14'
CLIP_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
LANGUAGE_DIM = 768
LANGUAGE_KEY = 'lang_emb'
PROMPT_SOURCES = ('task_name', 'task_description')
TOKENIZATION = 'clip_77_content_chunks_mean_v1'


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value


def select_prompts(dataset_path, task_map, prompt_source):
    if prompt_source not in PROMPT_SOURCES:
        raise ValueError(f'Unknown prompt_source {prompt_source!r}')
    if not task_map or any(type(index) is not int for index in task_map):
        raise ValueError('Need a nonempty integer task map')
    names = {index: _text(name, 'task_name') for index, name in task_map.items()}
    if len(set(names.values())) != len(names):
        raise ValueError('Duplicate selected task_name')
    if prompt_source == 'task_name':
        return [names[index] for index in sorted(names)]
    path = Path(dataset_path) / 'meta/tasks.jsonl'
    if not path.is_file():
        raise ValueError(f'task_description requires {path}')
    descriptions, seen_ids, seen_names = {}, set(), set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'Malformed {path}:{line_number}') from error
        if not isinstance(row, dict) or type(row.get('task_index')) is not int:
            raise ValueError(f'Invalid task_index in {path}:{line_number}')
        index = row['task_index']
        name = _text(row.get('task_name'), f'task_name in {path}:{line_number}')
        if index in seen_ids or name in seen_names:
            raise ValueError(f'Duplicate/conflicting task in {path}:{line_number}')
        seen_ids.add(index)
        seen_names.add(name)
        if index in names or name in names.values():
            if names.get(index) != name:
                raise ValueError(f'Conflicting task_name/task_index in {path}:{line_number}')
            descriptions[index] = _text(row.get('task'), f'Selected task description in {path}:{line_number}')
    missing = set(names) - set(descriptions)
    if missing:
        raise ValueError(f'Missing selected task descriptions: {sorted(missing)}')
    return [descriptions[index] for index in sorted(names)]


@torch.no_grad()
def encode_clip_prompts(prompts, *, tokenizer=None, model=None, revision=CLIP_REVISION):
    """Mean projected CLIP embeddings over lossless <=75-content-token chunks."""
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError('CLIP prompts must be a nonempty sequence of strings')
    if (tokenizer is None) != (model is None):
        raise ValueError('Supply both tokenizer and model, or neither')
    if model is None:
        try:
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
        except ImportError as error:
            raise RuntimeError('clip_film requires transformers>=4.46,<5; install requirements-b1k.txt') from error
        model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL, revision=revision)
        resolved = revision if re.fullmatch(r'[0-9a-f]{40}', revision) else getattr(model.config, '_commit_hash', None)
        if not resolved:
            from transformers import AutoConfig
            resolved = AutoConfig.from_pretrained(CLIP_MODEL, revision=revision)._commit_hash
        if not resolved:
            raise ValueError('CLIP model must have a resolved Hugging Face revision')
        tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL, revision=resolved)
    else:
        resolved = getattr(model.config, '_commit_hash', None) or revision
    model.eval().requires_grad_(False)
    if model.config.max_position_embeddings != 77 or model.config.projection_dim != LANGUAGE_DIM:
        raise ValueError('Expected CLIP ViT-L/14 text context 77 and projection dimension 768')
    device = next(model.parameters()).device
    embeddings = []
    for prompt in prompts:
        _text(prompt, 'prompt')
        content = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
        if len(content) <= 75:
            tokens = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        else:
            chunks = [content[start:start + 75] for start in range(0, len(content), 75)]
            input_ids = torch.full((len(chunks), 77), tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, chunk in enumerate(chunks):
                ids = [tokenizer.bos_token_id, *chunk, tokenizer.eos_token_id]
                input_ids[index, :len(ids)] = torch.tensor(ids)
                attention_mask[index, :len(ids)] = 1
        projected = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).text_embeds
        embeddings.append(projected.float().mean(dim=0).cpu())
    result = torch.stack(embeddings).detach()
    if result.shape != (len(prompts), LANGUAGE_DIM) or not torch.isfinite(result).all():
        raise ValueError('CLIP produced invalid/non-finite task embeddings')
    return result, resolved


def validate_language_cache(cache, task_map, prompt_source):
    if not isinstance(cache, dict) or cache.get('format') != 'clip_film_v1':
        raise ValueError('clip_film requires a checkpoint language cache')
    if prompt_source not in PROMPT_SOURCES or not task_map or any(type(index) is not int for index in task_map):
        raise ValueError('Invalid language prompt source or task map')
    ids = sorted(task_map)
    names = [_text(task_map[index], 'Cached task_name') for index in ids]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate cached task names')
    if (cache.get('prompt_source') != prompt_source or cache.get('model') != CLIP_MODEL or
            cache.get('tokenization') != TOKENIZATION or cache.get('task_ids') != ids or
            cache.get('task_names') != [task_map[index] for index in ids]):
        raise ValueError('Language cache source/model/task mapping does not match configuration')
    revision = _text(cache.get('revision'), 'CLIP revision')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('CLIP revision must be an exact 40-character commit hash')
    prompts = cache.get('prompts')
    if not isinstance(prompts, list) or len(prompts) != len(ids):
        raise ValueError('Language cache is missing selected prompts')
    for prompt in prompts:
        _text(prompt, 'Cached prompt')
    if prompt_source == 'task_name' and prompts != cache['task_names']:
        raise ValueError('Cached task_name prompts must equal raw metadata names')
    embeddings = cache.get('embeddings')
    if (not isinstance(embeddings, torch.Tensor) or embeddings.shape != (len(ids), LANGUAGE_DIM) or
            not embeddings.is_floating_point() or not torch.isfinite(embeddings).all()):
        raise ValueError('Language cache requires finite [tasks, 768] embeddings')
    embeddings = embeddings.detach().to(device='cpu', dtype=torch.float32).clone()
    if not torch.isfinite(embeddings).all():
        raise ValueError('Language cache embeddings must be finite in float32')
    return {**cache, 'embeddings': embeddings}


def prepare_language(dataset_path, task_map, language_conditioning='none', prompt_source='task_name', cached=None):
    if language_conditioning == 'none':
        if cached is not None:
            raise ValueError('Language cache supplied for language_conditioning=none')
        return None
    if language_conditioning != 'clip_film':
        raise ValueError(f'Unknown language_conditioning {language_conditioning!r}')
    if cached is None:
        prompts = select_prompts(dataset_path, task_map, prompt_source)
        # Model construction must not perturb diffusion initialization or resumed RNG state.
        with torch.random.fork_rng(devices=[]):
            embeddings, revision = encode_clip_prompts(prompts)
        cached = {'format': 'clip_film_v1', 'model': CLIP_MODEL, 'revision': revision,
                  'prompt_source': prompt_source, 'tokenization': TOKENIZATION,
                  'task_ids': sorted(task_map), 'task_names': [task_map[index] for index in sorted(task_map)],
                  'prompts': prompts, 'embeddings': embeddings}
    return validate_language_cache(cached, task_map, prompt_source)
