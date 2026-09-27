"""Version 1: 80 fixed additions evaluated in iterations 02 and 03."""
CANDIDATES = {
 'simple_smart': ['an intelligent-looking person','an unintelligent-looking person','a thoughtful person','a confused person','a clever-looking face','a dull-looking face','a knowledgeable-looking person','a vacant facial expression'],
 'simple_trustworthy': ['a trustworthy person','an untrustworthy person','a reassuring face','a suspicious face','an honest-looking person','a dishonest-looking person','a sincere person','a deceptive-looking person'],
 'simple_alert': ['an alert person','a tired person','a person with wide-open eyes','a person with sleepy eyes','an attentive face','a drowsy face','a focused facial expression','a distracted facial expression'],
 'simple_well-groomed': ['a well-groomed person','an unkempt person','a person with neatly styled hair','a person with messy hair','a neatly groomed face','a disheveled person','a person with neat eyebrows','a person with untidy facial hair'],
 'attention_cues': ['a person looking directly at the camera with focused eyes','a person gazing blankly into the distance','a person with raised upper eyelids and visible irises','a person with drooping upper eyelids','a person with narrowed eyes and furrowed eyebrows','a person with relaxed eyebrows and wide eyes','a person with a slack open mouth','a person with firmly closed lips','a person with their chin lifted and eyes looking down','a person with their chin lowered and eyes looking up','a person with dark circles under their eyes','a person with a tilted head and asymmetrical gaze'],
 'surface_presentation': ['a person with carefully shaped eyebrows','a person with bushy untidy eyebrows','a person with neatly trimmed facial hair','a person with patchy uneven facial hair','a person with smooth neatly combed hair','a person with flyaway tangled hair','a face with sharply defined makeup','a face with smudged makeup','a person with a polished formal appearance','a person with a casual disheveled appearance','a face with glossy oily skin','a face with dry rough skin'],
}
# First four simple captions are matched to the old four descriptor roles;
# remaining four expand that construct, so exact rewording gets a separate test.
for target,words in {'smart':('intelligent','unintelligent','thoughtful','confused'),
                     'trustworthy':('trustworthy','untrustworthy','reassuring','suspicious'),
                     'alert':('alert','tired','attentive','drowsy')}.items():
 CANDIDATES['conditional_'+target]=[f'a {word}-looking {person}' for word in words for person in ('man','woman')]

