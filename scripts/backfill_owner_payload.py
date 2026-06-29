"""Backfill ``owner_user_id`` onto Qdrant points using DB document ownership.

For collections seeded before per-owner attribution, points carry no
``owner_user_id`` payload and are invisible to owner-scoped retrieval. This
migration assigns each point its owner **from the database**: only points whose
document is assigned to a user (a ``documents`` row with ``owner_user_id`` set
and ``document_chunks.qdrant_point_id`` pointing at the point) are updated.
Orphaned points (no user assignment) are left untouched.

Because ownership comes from the DB, every backfilled point is guaranteed to
satisfy retrieval's PostgreSQL cross-check as well as the Qdrant filter.

Run where ``MEDIC_DATABASE_URL`` points at the target DB and ``QdrantURL`` at
the target collection (i.e. the production environment)::

    python -m scripts.backfill_owner_payload --dry-run   # report only
    python -m scripts.backfill_owner_payload             # apply
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from uuid import UUID

from qdrant_client.http import models
from sqlalchemy import select

from rag.database.models import Document, DocumentChunk
from rag.database.session import get_session_factory
from rag.qdrant import Qdrant

_OWNER_FIELD = "owner_user_id"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    qdrant = Qdrant(collection_name=args.collection)
    collection_name = qdrant.settings.qdrant_collection_name

    owner_to_points = _owned_points_by_owner()
    present_ids = _collection_point_ids(qdrant, collection_name)

    plan = _plan(owner_to_points, present_ids)
    _report(collection_name, plan, total_in_collection=len(present_ids))

    if not plan.updates:
        print("Nothing to do: no assigned documents map to points in this collection.")
        return 0
    if args.dry_run:
        print("Dry run: no changes written.")
        return 0

    _ensure_keyword_index(qdrant, collection_name)
    for owner, point_ids in plan.updates.items():
        qdrant.client.set_payload(
            collection_name=collection_name,
            payload={_OWNER_FIELD: str(owner)},
            points=sorted(point_ids),
            wait=True,
        )
    print(f"Updated {plan.points_to_update} point(s) across {len(plan.updates)} owner(s).")
    return 0


class _Plan:
    def __init__(self) -> None:
        self.updates: dict[UUID, set[str]] = {}
        self.assigned_total = 0
        self.missing_from_collection = 0

    @property
    def points_to_update(self) -> int:
        return sum(len(ids) for ids in self.updates.values())


def _plan(owner_to_points: dict[UUID, set[str]], present_ids: set[str]) -> _Plan:
    plan = _Plan()
    for owner, point_ids in owner_to_points.items():
        plan.assigned_total += len(point_ids)
        present = point_ids & present_ids
        plan.missing_from_collection += len(point_ids) - len(present)
        if present:
            plan.updates[owner] = present
    return plan


def _owned_points_by_owner() -> dict[UUID, set[str]]:
    session_factory = get_session_factory()
    statement = (
        select(Document.owner_user_id, DocumentChunk.qdrant_point_id)
        .join(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(
            Document.owner_user_id.is_not(None),
            DocumentChunk.qdrant_point_id.is_not(None),
        )
    )
    owner_to_points: dict[UUID, set[str]] = defaultdict(set)
    with session_factory() as session:
        for owner_user_id, qdrant_point_id in session.execute(statement):
            owner_to_points[UUID(str(owner_user_id))].add(str(qdrant_point_id))
    return dict(owner_to_points)


def _collection_point_ids(qdrant: Qdrant, collection_name: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        records, offset = qdrant.client.scroll(
            collection_name=collection_name,
            limit=256,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        ids.update(str(record.id) for record in records)
        if offset is None:
            return ids


def _ensure_keyword_index(qdrant: Qdrant, collection_name: str) -> None:
    qdrant.client.create_payload_index(
        collection_name=collection_name,
        field_name=_OWNER_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _report(collection_name: str, plan: _Plan, *, total_in_collection: int) -> None:
    print(f"collection                 : {collection_name}")
    print(f"points in collection       : {total_in_collection}")
    print(f"owners with assignments    : {len(plan.updates)}")
    print(f"assigned points (DB)       : {plan.assigned_total}")
    print(f"  -> present in collection : {plan.points_to_update}")
    print(f"  -> missing from collection: {plan.missing_from_collection}")
    print(f"unassigned points (orphans): {total_in_collection - plan.points_to_update}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="backfill_owner_payload")
    parser.add_argument("--collection", default=None, help="Override collection name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any payload",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
