"""The scorer decides every "correct" in the benchmark, so it is tested first.

It follows the official GAIA quasi-exact-match logic, including the parts that
are stricter than they look: only "$", "%" and "," are stripped from numbers.
"""

import pytest

from bench.scorer import extract_final_answer, question_scorer


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("1,234", "1234"),            # thousands separator
        ("$1,234.50", "1234.5"),      # currency sign and comma
        ("42%", "42"),                # percent sign
        ("3.0", "3"),                 # float against integer
        ("0.50", "0.5"),              # trailing zero
        (" 17 ", "17"),               # stray whitespace around a number
        ("-12", "-12"),
    ],
)
def test_numbers_match_after_normalisation(answer, truth):
    assert question_scorer(answer, truth)


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("3.14159", "3.14"),          # close is not equal
        ("12 kg", "12"),              # word units are NOT stripped upstream
        ("twelve", "12"),             # words are not numbers
        ("", "12"),
        ("1234", "1235"),
    ],
)
def test_numbers_that_must_not_match(answer, truth):
    assert not question_scorer(answer, truth)


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("Paris", "paris"),                       # case
        ("  New   York ", "new york"),            # extra whitespace
        ("St. Petersburg", "St Petersburg"),      # punctuation in a plain string
        ("velvet", "Velvet"),
    ],
)
def test_strings_match_case_and_whitespace_insensitively(answer, truth):
    assert question_scorer(answer, truth)


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("London", "Paris"),
        ("New York City", "New York"),
        ("", "Paris"),
    ],
)
def test_strings_that_must_not_match(answer, truth):
    assert not question_scorer(answer, truth)


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("a, b, c", "a,b,c"),                     # spacing inside a list
        ("Apple; Banana", "apple;banana"),        # semicolon lists, case
        ("1, 2.0, 3", "1,2,3"),                   # numeric elements compared as numbers
        ("red, 5, Blue", "Red,5,blue"),           # mixed list
    ],
)
def test_lists_match_element_by_element(answer, truth):
    assert question_scorer(answer, truth)


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("a, b", "a,b,c"),                        # too short
        ("a, b, c, d", "a,b,c"),                  # too long
        ("b, a, c", "a,b,c"),                     # order matters
        ("a, b, x", "a,b,c"),                     # one wrong element
        ("1, 2, 4", "1,2,3"),
    ],
)
def test_lists_that_must_not_match(answer, truth):
    assert not question_scorer(answer, truth)


def test_a_missing_answer_never_scores():
    assert not question_scorer(None, "42")
    assert not question_scorer(None, "Paris")


def test_final_answer_is_taken_from_the_marker():
    answer, had_marker = extract_final_answer("I reasoned a lot.\nFINAL ANSWER: 42")

    assert (answer, had_marker) == ("42", True)


def test_the_last_marker_wins():
    text = "FINAL ANSWER: draft\nActually no.\nFINAL ANSWER: Paris"

    assert extract_final_answer(text)[0] == "Paris"


def test_marker_is_found_whatever_its_case():
    assert extract_final_answer("final answer: 7") == ("7", True)


def test_without_a_marker_the_whole_message_is_the_answer():
    assert extract_final_answer("  Paris  ") == ("Paris", False)
