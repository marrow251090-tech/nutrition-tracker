
"""Supabase repository for the shared food catalog."""

from dataclasses import dataclass

from supabase import Client


@dataclass
class SupabaseFoodCatalogRepository:
    """Read foods from the shared food catalog."""

    client: Client

    def search_foods(
        self, query: str, limit: int = 5
    ) -> list[dict[str, object]]:
        """Search shared foods by normalized Russian name."""
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        response = (
            self.client.table("food_catalog")
            .select(
                "id,name,name_normalized,category,basis,"
                "calories,protein_g,fat_g,carbs_g,source"
            )
            .ilike("name_normalized", f"%{normalized_query}%")
            .limit(limit)
            .execute()
        )

        return [dict(row) for row in (response.data or [])]

    def get_food_by_name(
        self, query: str
    ) -> dict[str, object] | None:
        """Find one shared food by its normalized name."""
        results = self.search_foods(query, limit=10)
        if not results:
            return None

        exact_name = query.strip().lower()

        for food in results:
            if food.get("name_normalized") == exact_name:
                return food

        return results[0]
