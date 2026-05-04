from __future__ import annotations

from collections.abc import Callable

from evaluation.application.models import DatasetBootstrapResult
from evaluation.application.ports import DatasetBootstrapGateway, ProfileRepository


class BootstrapEvaluationDataset:
    def __init__(
        self,
        *,
        profiles: ProfileRepository,
        experiments: DatasetBootstrapGateway,
        manifest_path_for: Callable[[str], str],
    ) -> None:
        self._profiles = profiles
        self._experiments = experiments
        self._manifest_path_for = manifest_path_for

    def execute(self, profile_id: str) -> DatasetBootstrapResult:
        profile = self._profiles.get(profile_id)
        self._experiments.authenticate()
        return self._experiments.bootstrap(
            profile=profile,
            manifest_path=self._manifest_path_for(profile_id),
        )
