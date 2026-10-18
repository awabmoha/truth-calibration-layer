"""Unit tests for labeling rule v3. Run: python tcl_experiments/scripts/test_labeling_v3.py"""

from labeling import is_correct, is_correct_v3, normalize_tokens_v3

# (answer, accepted answers, expected v3 label)
CASES = [
    # Form-level errors of rule v2 found in the Phase 1 label audit
    ("pathogens", ["pathogen attack", "pathogen"], 1),
    ("recover the latent heat of vaporization", ["recover the latent heat of vaporisation"], 1),
    ("mid-sixties", ["the mid-sixties", "1965"], 1),
    ("Oxygen mask", ["mechanical ventilators", "oxygen masks"], 1),
    ("bone", ["bones"], 1),
    ("United Kingdom, Australia, Canada, United States", ["the United Kingdom, Australia, Canada and the United States"], 1),
    ("Gamal Abdel Nasser", ["Gamal Abdul Nasser"], 1),
    ("4.5", ["Four", "4"], 0),
    ("About 6.04 mL of oxygen per liter of fresh water", ["6.04 milliliters"], 1),
    ("The Doctor Who DVDs are available in the UK, Australia, Canada and the United States.",
     ["the United Kingdom, Australia, Canada and the United States"], 1),
    ("The context does not specify whether atmospheric oxygen levels are going up, down, or staying the same",
     ["down", "downward"], 0),
    ("The context does not specify which is older, the British Empire or the Ethiopian Empire",
     ["Ethiopian Empire", "Ethiopian"], 0),
    ("According to the context, an oxygen mask in medical applications can be a concern",
     ["mechanical ventilators", "oxygen masks"], 1),
    ("1.5 meters", ["1.5 m (4 ft 11 in)", "a few millimeters to 1.5 m"], 1),
    ("The Islamic Liberation Party has attempted to assassinate political figures. The context does not specify the names.",
     ["political figures"], 1),
    ("German Nazis", ["German Nazi"], 1),
    ("6.0 points", ["6", "six"], 1),
    ("Montezuma II", ["Moctezuma II"], 1),
    ("copper and zinc", ["copper (Cu)"], 1),
    ("The President of the United States", ["the President"], 1),
    ("100 Celsius", ["100 °C"], 1),
    ("100 degrees Celsius", ["100 °C"], 1),
    ("3600 RPM", ["3600 revolutions per minute"], 1),
    ("12-15 million", ["12 to 15 million"], 1),
    ("300 kilometers", ["300 km long"], 1),
    ("The Fifth, Sixth and Seventh Doctors were highlighted", ["the Fifth, Sixth and Seventh Doctors"], 1),
    ("2,000", ["over 2,000", "2,000 buildings"], 1),
    # Traps that must stay incorrect
    ("1963", ["1963 to 1989"], 0),
    ("M. Smith", ["meter"], 0),
    ("Golden Retriever", ["Retriever (Labrador)", "Labrador Retriever"], 0),
    ("The art is called batmaking.", ["Bat (cricket)"], 0),
    ("Northeastern UK", ["Northern United Kingdom"], 0),
    ("Austria", ["Australia"], 0),
    ("Louis XIV", ["Louis XVI"], 0),
    ("Radar", ["Racecar"], 0),
    ("Henry VIII", ["Henry"], 0),
    ("The answer is not provided in the context, so I cannot answer this question.", ["context"], 0),
    ("14", ["4"], 0),
    ("1912", ["1914"], 0),
    # Cases rule v2 already handled, which v3 must keep
    ("+49", ["49", "forty-nine"], 1),
    ("Racecar", ["Racecar"], 1),
    ("1914", ["one thousand, nine hundred and fourteen", "1914"], 1),
    ("The answer is South Africa.", ["South Africa"], 1),
    ("Dr. Pangloss", ["Dr Pangloss", "Pangloss"], 1),
    ("Bogotá", ["Bogota"], 1),
    ("The Gunpowder Plot of 1605", ["GUNPOWDER PLOT"], 1),
    ("Velvet Revolution", ["The Velvet Revolution"], 1),
    ("Three strokes under par", ["Three", "3"], 1),
    ("Saturn", ["Saturn"], 1),
    ("the U.S.", ["United States"], 1),
    ("colours", ["color"], 1),
    ("29.7%", ["29.7%"], 1),
    ("11.8%", ["29.7%"], 0),
]


# (answer, accepted answers, question, expected v3 label): cases that depend on the question
QUESTION_CASES = [
    ("A bairn is a small mound or hill.", ["Child", "Bairns"], "In Scotland, what is a bairn?", 0),
    ("The title of the play 'Die Rauber' is", ["The Robbers", "Die Raeuber"],
     "What does the title of the play 'Die Rauber' translate to in English?", 0),
    ("A child", ["Child", "Bairns"], "In Scotland, what is a bairn?", 1),
    ("Jaffa cakes are a cake.", ["Cake", "cakes"], "is jaffa cakes a biscuit or a cake", 1),
    ("going down", ["down", "downward"], "Are atmospheric oxygen levels going up, down, or staying the same?", 1),
    ("18 shots", ["18", "eighteen"], "What is the least number of shots in a round of 18 holes?", 1),
    ("The answer is 4.", ["4", "four"], "Steffi Graf won each of the 4 Grand Slam events at least 4 times?", 1),
    ("Mercury is the working fluid in a mercury vapor turbine.", ["Mercury"],
     "What working fluid is used in a mercury vapor turbine?", 1),
    ("Charlie Sheen as Charlie Harper", ["Charlie"], "Charlie Sheen starred as which character in the TV series?", 1),
    ("The answer is snapdragon.", ["Snapdragon", "Antirrhinum"], "What is another name for the snapdragon?", 0),
    ("ten thousand", ["Googol", "Ten thousand sexdecillion"], "Ten raised to the power of one hundred is called what?", 0),
]


def main() -> None:
    failures = 0
    for answer, accepted, question, expected in QUESTION_CASES:
        got = int(is_correct_v3(answer, accepted, question))
        if got != expected:
            failures += 1
            print(f"FAIL v3={got} expected={expected} | {answer!r} vs {accepted} | Q: {question}")
    for answer, accepted, expected in CASES:
        got = int(is_correct_v3(answer, accepted, "question"))
        if got != expected:
            failures += 1
            print(f"FAIL v3={got} expected={expected} | {answer!r} vs {accepted}")
            print(f"     tokens: {normalize_tokens_v3(answer)} vs {[normalize_tokens_v3(g) for g in accepted]}")
    v2_fixed = sum(int(is_correct(a, g, "question")) != e and int(is_correct_v3(a, g, "question")) == e for a, g, e in CASES)
    print(f"{len(CASES) - failures}/{len(CASES)} cases pass; v3 fixes {v2_fixed} cases that v2 gets wrong")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
