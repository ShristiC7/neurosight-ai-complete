"""
NeuroSight AI — Historical/Batch Analytics Service
Handles database queries and aggregation for long-term trends and focus heatmaps.
"""
import uuid
from datetime import datetime, date, timezone, timedelta
from sqlalchemy import select, and_, func, Date
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import WorkSession, BehavioralMetric
from app.core.redis import redis_client
import json

class HistoricalAnalyticsService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_batch_analytics(
        self, user_id: uuid.UUID, start_date: date, end_date: date, period: str = "daily"
    ) -> dict:
        """
        Retrieves aggregated performance metrics (fatigue, stress, productivity, focus time)
        for a user over a date range, grouped by daily or weekly periods.
        """
        # Convert start/end dates to datetime with UTC timezone
        start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

        if period == "weekly":
            # Group by week start
            group_expr = func.date_trunc('week', WorkSession.start_time)
        else:
            # Default to daily grouping
            group_expr = func.cast(WorkSession.start_time, Date)

        stmt = (
            select(
                group_expr.label("period_label"),
                func.avg(WorkSession.avg_fatigue_score).label("avg_fatigue"),
                func.avg(WorkSession.avg_stress_score).label("avg_stress"),
                func.avg(WorkSession.avg_productivity_score).label("avg_productivity"),
                func.sum(WorkSession.total_focus_time).label("total_focus"),
                func.count(WorkSession.id).label("sessions_count")
            )
            .where(
                and_(
                    WorkSession.user_id == user_id,
                    WorkSession.start_time >= start_dt,
                    WorkSession.start_time <= end_dt
                )
            )
            .group_by(group_expr)
            .order_by(group_expr.asc())
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        data_points = []
        for r in rows:
            period_val = r.period_label
            if isinstance(period_val, datetime):
                period_val = period_val.date()

            data_points.append({
                "period": period_val.isoformat() if period_val else "",
                "avg_fatigue_score": round(float(r.avg_fatigue or 0.0), 1),
                "avg_stress_score": round(float(r.avg_stress or 0.0), 1),
                "avg_productivity_score": round(float(r.avg_productivity or 0.0), 1),
                "total_focus_time_minutes": int(r.total_focus or 0),
                "sessions_count": int(r.sessions_count or 0)
            })

        # Calculate overall summary stats for this period
        summary_stmt = (
            select(
                func.avg(WorkSession.avg_fatigue_score).label("avg_fatigue"),
                func.avg(WorkSession.avg_stress_score).label("avg_stress"),
                func.avg(WorkSession.avg_productivity_score).label("avg_productivity"),
                func.sum(WorkSession.total_focus_time).label("total_focus"),
                func.count(WorkSession.id).label("sessions_count")
            )
            .where(
                and_(
                    WorkSession.user_id == user_id,
                    WorkSession.start_time >= start_dt,
                    WorkSession.start_time <= end_dt
                )
            )
        )
        summary_result = await self.db.execute(summary_stmt)
        s_row = summary_result.one()

        summary = {
            "avg_fatigue_score": round(float(s_row.avg_fatigue or 0.0), 1),
            "avg_stress_score": round(float(s_row.avg_stress or 0.0), 1),
            "avg_productivity_score": round(float(s_row.avg_productivity or 0.0), 1),
            "total_focus_time_minutes": int(s_row.total_focus or 0),
            "total_sessions": int(s_row.sessions_count or 0)
        }

        return {
            "user_id": str(user_id),
            "start_date": start_date,
            "end_date": end_date,
            "period": period,
            "data_points": data_points,
            "summary": summary
        }

    async def get_focus_heatmap(self, user_id: uuid.UUID) -> list[dict]:
        """
        Retrieves the focus heatmap data for a user.
        Attempts to read from Redis cache first, falling back to database aggregation if missing.
        """
        cache_key = f"neurosight:heatmap:{user_id}"
        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception:
            pass

        # Database fallback
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        result = await self.db.execute(
            select(BehavioralMetric)
            .where(
                and_(
                    BehavioralMetric.user_id == user_id,
                    BehavioralMetric.timestamp >= cutoff
                )
            )
        )
        metrics = result.scalars().all()

        grid: dict[str, list[float]] = {}
        for m in metrics:
            ts = m.timestamp.replace(tzinfo=timezone.utc)
            key = f"{ts.weekday()}-{ts.hour}"
            grid.setdefault(key, []).append(m.behavior_score)

        heatmap = []
        for day in range(7):
            for hour in range(24):
                key = f"{day}-{hour}"
                scores = grid.get(key, [])
                avg = sum(scores) / len(scores) if scores else 0.0
                heatmap.append({"day": day, "hour": hour, "value": round(avg, 1)})

        # Cache in Redis async
        try:
            await redis_client.setex(cache_key, 3600, json.dumps(heatmap))
        except Exception:
            pass

        return heatmap
