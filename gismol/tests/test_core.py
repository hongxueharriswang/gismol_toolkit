
import pytest
from gismol import COH

def test_hierarchy_cycle():
    a = COH(name='A')
    b = COH(name='B')
    a.add_child(b)
    with pytest.raises(ValueError):
        b.add_child(a)  # would create cycle
