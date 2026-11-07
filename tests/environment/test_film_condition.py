import pytest
from src.environment.stimulus import FilmCondition, Stimulus


@pytest.fixture
def film_condition():
    """Fixture for instance of film condition"""
    return FilmCondition(0.01, 293.0, all_elements=True)

def test_id_increment():
    """test the id of constraints increments with each initialisation"""
    Stimulus.reset_instance_count()
    constraint_0 = FilmCondition(0.01, 293.0, all_elements=True)
    constraint_1 = FilmCondition(pos_x=0.0, pos_y=0.0, pos_z=0.0, film_coefficient=0.01, ambient_temperature=293.0)
    assert constraint_0.id == 0
    assert constraint_1.id == 1

def test_film_condition_is_support(film_condition):
    """Test film condition is an instance of the Stimulus class"""
    assert isinstance(film_condition, Stimulus)

def test_inp_all_elements(film_condition):
    inp_deck = film_condition.get_inp_deck()
    assert inp_deck == '*FILM\nALLELEMENTS, F2, 293.0, 0.01\n'
