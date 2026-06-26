from evaluation.application.errors import CorpusIsolationError


class EvaluationCollectionGuard:
    def __init__(self, production_collection_name: str) -> None:
        self._production_collection_name = production_collection_name

    def validate(self, collection_name: str) -> None:
        if not collection_name.startswith("medic_eval_"):
            raise CorpusIsolationError(
                "Evaluation collection name must start with 'medic_eval_'"
            )
        if collection_name == self._production_collection_name:
            raise CorpusIsolationError(
                "Evaluation collection cannot be the production collection"
            )
