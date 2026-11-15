"""What each column of student-mat.csv means, in plain words.

Every column belongs to one of the four kinds of information the project
brief names: demographic, academic, social and behavioural. The app uses
`question` as the input label and `choices` to show words instead of codes.
Source: attribute list on the UCI dataset page.
"""

from dataclasses import dataclass

GROUPS = ("Demographic", "Academic", "Social", "Behavioural")


@dataclass(frozen=True)
class Attribute:
    name: str                    # column name in the CSV
    group: str                   # one of GROUPS
    question: str                # label shown in the app
    choices: dict | None = None  # CSV value -> wording; None for plain numbers


ONE_TO_FIVE = {1: "1 - very low", 2: "2 - low", 3: "3 - moderate", 4: "4 - high", 5: "5 - very high"}
QUALITY = {1: "1 - very poor", 2: "2 - poor", 3: "3 - fair", 4: "4 - good", 5: "5 - excellent"}
SCHOOLING = {0: "No schooling", 1: "Up to 4th grade", 2: "5th-9th grade",
             3: "Secondary school", 4: "University"}
OCCUPATION = {"teacher": "Teacher", "health": "Health sector", "services": "Public services",
              "at_home": "Stays at home", "other": "Something else"}
YES_NO = {"yes": "Yes", "no": "No"}

ATTRIBUTES = [
    # Demographic
    Attribute("school", "Demographic", "School attended",
              {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}),
    Attribute("sex", "Demographic", "Student's sex", {"F": "Female", "M": "Male"}),
    Attribute("age", "Demographic", "Age in years"),
    Attribute("address", "Demographic", "Lives in", {"U": "A town or city", "R": "The countryside"}),
    Attribute("famsize", "Demographic", "Household size",
              {"LE3": "Three people or fewer", "GT3": "More than three people"}),
    Attribute("Pstatus", "Demographic", "Parents' living arrangement",
              {"T": "Together", "A": "Separately"}),
    Attribute("Medu", "Demographic", "Mother's schooling", SCHOOLING),
    Attribute("Fedu", "Demographic", "Father's schooling", SCHOOLING),
    Attribute("Mjob", "Demographic", "Mother's occupation", OCCUPATION),
    Attribute("Fjob", "Demographic", "Father's occupation", OCCUPATION),
    Attribute("guardian", "Demographic", "Legal guardian",
              {"mother": "Mother", "father": "Father", "other": "Someone else"}),
    Attribute("traveltime", "Demographic", "Journey to school",
              {1: "Under 15 minutes", 2: "15-30 minutes", 3: "30-60 minutes", 4: "Over an hour"}),
    Attribute("internet", "Demographic", "Internet at home", YES_NO),
    # Academic
    Attribute("G1", "Academic", "First-term maths grade (0-20)"),
    Attribute("G2", "Academic", "Second-term maths grade (0-20)"),
    Attribute("failures", "Academic", "Classes failed before (3 = three or more)"),
    Attribute("studytime", "Academic", "Study hours per week",
              {1: "Less than 2", 2: "2 to 5", 3: "5 to 10", 4: "More than 10"}),
    Attribute("higher", "Academic", "Plans to go to university", YES_NO),
    Attribute("reason", "Academic", "Main reason for picking this school",
              {"home": "Near home", "reputation": "Its reputation",
               "course": "The courses", "other": "Another reason"}),
    Attribute("schoolsup", "Academic", "Gets extra help from the school", YES_NO),
    Attribute("famsup", "Academic", "Gets study help from family", YES_NO),
    Attribute("paid", "Academic", "Takes paid maths tuition", YES_NO),
    Attribute("nursery", "Academic", "Went to nursery school", YES_NO),
    # Social
    Attribute("famrel", "Social", "Relationship with family", QUALITY),
    Attribute("freetime", "Social", "Free time after school", ONE_TO_FIVE),
    Attribute("goout", "Social", "Time out with friends", ONE_TO_FIVE),
    Attribute("romantic", "Social", "Has a boyfriend or girlfriend", YES_NO),
    Attribute("activities", "Social", "Does after-school activities", YES_NO),
    # Behavioural
    Attribute("absences", "Behavioural", "Days absent from school"),
    Attribute("Dalc", "Behavioural", "Drinking on school days", ONE_TO_FIVE),
    Attribute("Walc", "Behavioural", "Drinking at weekends", ONE_TO_FIVE),
    Attribute("health", "Behavioural", "Health right now", QUALITY),
]

BY_NAME = {attribute.name: attribute for attribute in ATTRIBUTES}
