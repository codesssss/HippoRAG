import json
from typing import Dict, Tuple

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo, build_causal_fact_records, parse_causal_relations_response
from ..utils.misc_utils import CausalRawOutput, NerRawOutput, TripleRawOutput
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager
from ..llm.vllm_offline import VLLMOffline
from ..utils.llm_utils import filter_invalid_triples

logger = get_logger(__name__)


def _sanitize_named_entities(unique_entities):
    return [entity for entity in unique_entities if isinstance(entity, str)]


class VLLMOfflineOpenIE(OpenIE):
    def __init__(self, global_config):

        self.global_config = global_config
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = VLLMOffline(global_config)

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, CausalRawOutput]]:
        """
        Conduct batch OpenIE synchronously using vLLM offline batch mode, including NER and triple extraction

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

        ner_input_messages = [self.prompt_template_manager.render(name='ner', passage=p) for p in chunk_passages.values()]
        ner_output, ner_output_metadata = self.llm_model.batch_infer(ner_input_messages, json_template='ner', max_tokens=512)

        triple_extract_input_messages = [self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=named_entities
        ) for passage, named_entities in zip(chunk_passages.values(), ner_output)]
        triple_output, triple_output_metadata = self.llm_model.batch_infer(triple_extract_input_messages, json_template='triples', max_tokens=2048)

        ner_raw_outputs = []
        for idx, ner_output_instance in enumerate(ner_output):
            chunk_id = list(chunks.keys())[idx]
            response = ner_output_instance
            try:
                unique_entities = _sanitize_named_entities(json.loads(response)["named_entities"])
            except Exception as e:
                unique_entities = []
                logger.warning(f"Could not parse response from OpenIE: {e}")
            if len(unique_entities) == 0:
                logger.warning("No entities extracted for chunk_id: {}".format(chunk_id))
            ner_raw_output = NerRawOutput(chunk_id, response, unique_entities, {})
            ner_raw_outputs.append(ner_raw_output)
        ner_results_dict = {chunk_key: ner_raw_output for chunk_key, ner_raw_output in zip(chunks.keys(), ner_raw_outputs)}

        triple_raw_outputs = []
        for idx, triple_output_instance in enumerate(triple_output):
            chunk_id = list(chunks.keys())[idx]
            response = triple_output_instance
            try:
                triples = filter_invalid_triples(json.loads(response)["triples"])
            except Exception as e:
                triples = []
                logger.warning(f"Could not parse response from OpenIE: {e}")
            if len(triples) == 0:
                logger.warning("No triples extracted for chunk_id: {}".format(chunk_id))
            triple_raw_output = TripleRawOutput(chunk_id, response, triples, {})
            triple_raw_outputs.append(triple_raw_output)
        triple_results_dict = {chunk_key: triple_raw_output for chunk_key, triple_raw_output in zip(chunks.keys(), triple_raw_outputs)}

        causal_raw_outputs = []
        if not getattr(self.global_config, "causal_enabled", True):
            for chunk_id in chunks.keys():
                causal_raw_outputs.append(CausalRawOutput(chunk_id, "", [], {}))
        else:
            causal_extract_input_messages = [
                self.prompt_template_manager.render(
                    name="causal_extraction",
                    passage=chunk_passages[triple_raw_output.chunk_id],
                    fact_records_json=json.dumps(build_causal_fact_records(triple_raw_output.triples)),
                )
                for triple_raw_output in triple_raw_outputs
            ]
            causal_output, causal_output_metadata = self.llm_model.batch_infer(
                causal_extract_input_messages,
                json_template="causal_relations",
                max_tokens=1024,
            )

            confidence_threshold = getattr(self.global_config, "causal_confidence_threshold", 0.5)
            for idx, causal_output_instance in enumerate(causal_output):
                chunk_id = list(chunks.keys())[idx]
                response = causal_output_instance
                fact_records = build_causal_fact_records(triple_raw_outputs[idx].triples)
                causal_relations = parse_causal_relations_response(
                    real_response=response,
                    valid_fact_ids=[record["fact_id"] for record in fact_records],
                    confidence_threshold=confidence_threshold,
                )
                causal_raw_outputs.append(CausalRawOutput(chunk_id, response, causal_relations, {}))
        causal_results_dict = {chunk_key: causal_raw_output for chunk_key, causal_raw_output in zip(chunks.keys(), causal_raw_outputs)}

        return ner_results_dict, triple_results_dict, causal_results_dict
