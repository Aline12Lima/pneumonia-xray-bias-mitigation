"""Dataset de modelos: lista das 15 execuções finais."""
from src.cache.modelos import execucoes


def test_quinze_execucoes_finais():
    ex = execucoes()
    assert len(ex) == 15 and len(set(ex)) == 15
    assert "M1_s0_lr0.0001" in ex and "M2_s4_lr0.0003" in ex and "M3_s2_lr0.0001" in ex
