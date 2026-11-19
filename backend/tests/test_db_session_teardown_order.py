"""Regression for #188: full-suite local deadlock.

``db_session``'s per-test ``drop_all`` must run *after* the global
``app.database.engine`` is disposed, not before. If it runs first, a
lingering own-session connection (patient_timeline & co., which use
``app.database.async_session_maker`` directly) can still hold locks that
block the ``DROP TABLE`` — the deadlock reported in #188.

This doesn't reproduce the deadlock itself (timing/protocol-dependent
across event loops, and per #188 doesn't reliably reproduce even in CI).
Instead it asserts the concrete, controllable invariant the fix provides:
``engine.dispose()`` is called before ``Base.metadata.drop_all`` during
``db_session`` teardown.

The assertion lives in the spy fixture's own teardown, which runs after
``db_session``'s teardown (autouse fixtures are set up first, so torn down
last). That keeps the test self-contained: it must not depend on another
test having run first, because CI splits tests across jobs by node id.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import app.database as database_module


@pytest.fixture(autouse=True)
def _spy_dispose_and_drop_all(monkeypatch):
    """Record dispose()/drop_all() order, then assert it once db_session is gone."""
    order: list[str] = []
    engine_cls = type(database_module.engine)
    real_dispose = engine_cls.dispose

    async def spy_dispose(self, *args, **kwargs):
        order.append("dispose")
        return await real_dispose(self, *args, **kwargs)

    monkeypatch.setattr(engine_cls, "dispose", spy_dispose)

    real_drop_all = database_module.Base.metadata.drop_all

    def spy_drop_all(bind, **kwargs):
        order.append("drop_all")
        return real_drop_all(bind, **kwargs)

    monkeypatch.setattr(database_module.Base.metadata, "drop_all", spy_drop_all)

    yield

    # Runs after db_session's teardown (LIFO), so both calls are recorded.
    # Pre-fix: only the autouse ``_dispose_app_engine`` fixture disposes the
    # engine, and it is torn down last, so drop_all is recorded before dispose.
    # Post-fix: db_session disposes explicitly first.
    assert "dispose" in order, "engine.dispose() was never called"
    assert "drop_all" in order, "Base.metadata.drop_all() was never called"
    assert order.index("dispose") < order.index("drop_all"), (
        f"dispose must happen before drop_all, got order: {order}"
    )


@pytest.mark.asyncio
async def test_db_session_disposes_engine_before_drop_all(db_session: AsyncSession) -> None:
    """One full db_session setup/teardown cycle; the order is asserted in the fixture."""
    assert db_session is not None
