import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.causal_utils import sanitize_causal_relations
from ..utils.misc_utils import (
    CausalRawOutput,
    CausalRelation,
    TripleRawOutput,
    NerRawOutput,
    compute_fact_id,
)
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    payload = match.group()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(payload)
    named_entities = parsed.get("named_entities", [])
    # Some local models emit placeholders like `...`; drop anything non-string
    # instead of crashing later when serializing entities back to JSON.
    return [entity for entity in named_entities if isinstance(entity, str)]


def _sanitize_named_entities(named_entities: List[Any]) -> List[str]:
    return [entity for entity in named_entities if isinstance(entity, str)]


def build_causal_fact_records(triples: List[List[str]]) -> List[Dict[str, Any]]:
    fact_records = []
    for triple in filter_invalid_triples(triples):
        fact_records.append(
            {
                "fact_id": compute_fact_id(triple),
                "triple": [str(item) for item in triple],
            }
        )
    return fact_records


def parse_causal_relations_response(real_response: str,
                                    valid_fact_ids: List[str],
                                    confidence_threshold: float) -> List[CausalRelation]:
    try:
        parsed = json.loads(real_response)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(real_response)
        except (ValueError, SyntaxError):
            parsed = {}

    raw_relations = parsed.get("causal_relations", []) if isinstance(parsed, dict) else []
    return sanitize_causal_relations(
        raw_relations=raw_relations,
        valid_fact_ids=valid_fact_ids,
        confidence_threshold=confidence_threshold,
    )


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, triple_extraction_template: str = "triple_extraction"):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.global_config = getattr(llm_model, "global_config", None)
        self.triple_extraction_template = triple_extraction_template

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
                response_format={"type": "json_object"},
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                # If pattern doesn't match, return an empty list
                return []
            payload = match.group()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(payload)
            return parsed.get("triples", [])

        raw_response = ""
        metadata = {}
        try:
            sanitized_entities = _sanitize_named_entities(named_entities)
            # PREPROCESSING
            messages = self.prompt_template_manager.render(
                name=self.triple_extraction_template,
                passage=passage,
                named_entity_json=json.dumps({"named_entities": sanitized_entities})
            )
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                response_format={"type": "json_object"},
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def causal_relation_extraction(self, chunk_key: str, passage: str, triples: List[List[str]]) -> CausalRawOutput:
        if self.global_config is not None and not getattr(self.global_config, "causal_enabled", True):
            return CausalRawOutput(
                chunk_id=chunk_key,
                response="",
                causal_relations=[],
                metadata={},
            )

        fact_records = build_causal_fact_records(triples)
        if not fact_records:
            return CausalRawOutput(
                chunk_id=chunk_key,
                response="",
                causal_relations=[],
                metadata={},
            )

        raw_response = ""
        metadata = {}
        try:
            messages = self.prompt_template_manager.render(
                name="causal_extraction",
                passage=passage,
                fact_records_json=json.dumps(fact_records),
            )
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                response_format={"type": "json_object"},
            )
            metadata["cache_hit"] = cache_hit
            if metadata.get("finish_reason") == "length":
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            causal_relations = parse_causal_relations_response(
                real_response=real_response,
                valid_fact_ids=[record["fact_id"] for record in fact_records],
                confidence_threshold=getattr(self.global_config, "causal_confidence_threshold", 0.5),
            )
        except Exception as e:
            logger.warning(f"Exception for causal extraction on chunk {chunk_key}: {e}")
            metadata.update({"error": str(e)})
            return CausalRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                causal_relations=[],
            )

        return CausalRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            causal_relations=causal_relations,
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        causal_output = self.causal_relation_extraction(chunk_key=chunk_key, passage=passage, triples=triple_output.triples)
        return {"ner": ner_output, "triplets": triple_output, "causal_relations": causal_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, CausalRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, CausalRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
                - A dict with keys as the chunk ids and values as the causal extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor() as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor() as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        causal_results_list = []
        if self.global_config is not None and not getattr(self.global_config, "causal_enabled", True):
            causal_results_list = [
                CausalRawOutput(chunk_id=res.chunk_id, response="", causal_relations=[], metadata={})
                for res in triple_results_list
            ]
        else:
            total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
            with ThreadPoolExecutor() as executor:
                causal_futures = {
                    executor.submit(
                        self.causal_relation_extraction,
                        triple_result.chunk_id,
                        chunk_passages[triple_result.chunk_id],
                        triple_result.triples,
                    ): triple_result.chunk_id
                    for triple_result in triple_results_list
                }
                pbar = tqdm(as_completed(causal_futures), total=len(causal_futures), desc="Extracting causal relations")
                for future in pbar:
                    result = future.result()
                    causal_results_list.append(result)
                    metadata = result.metadata
                    total_prompt_tokens += metadata.get("prompt_tokens", 0)
                    total_completion_tokens += metadata.get("completion_tokens", 0)
                    if metadata.get("cache_hit"):
                        num_cache_hit += 1
                    pbar.set_postfix({
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit
                    })

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}
        causal_results_dict = {res.chunk_id: res for res in causal_results_list}

        return ner_results_dict, triple_results_dict, causal_results_dict
