#!/usr/bin/env python3
"""
v5 Training Data Preparation: 80% Baseline + 20% Hard Cases & Contrastive Pairs.
Creates data_v5/train.csv and data_v5/val.csv.
"""

import csv
import json
import os
import random
import re
from datasets import load_dataset

random.seed(42)

# Output directory
DATA_V5_DIR = "/home/ubuntu/Content-Moderation-Model/data_v5"
os.makedirs(DATA_V5_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Strict Held-out Sets (NEVER leak into train or val)
# ---------------------------------------------------------------------------
BATTERY_HISTORICAL_24 = [
    # Benign garbage / figurative (14 cases)
    "Please put the garbage in the trash",
    "Please take out the garbage",
    "Take the garbage out to the curb before sunrise",
    "Sorting garbage and recycling is everyone's job",
    "He works as a garbage collector in our city",
    "The garbage truck comes on Monday",
    "Trash pickup in our neighborhood is on Fridays",
    "My phone battery is trash",
    "The trash can is full, someone empty it",
    "Garbage collection is on Tuesdays this week",
    "I love this post, very well written",
    "I could kill for a cup of coffee right now",
    "Her presentation killed it, everyone loved it",
    "He was niggardly with his praise",
    # Toxic direct (10 cases)
    "You are totally trash",
    "You are trash",
    "You are a worthless piece of garbage",
    "Nobody likes you, just leave",
    "Shut up you idiot",
    "Assholes",
    "NIGGER",
    "You are such a scumbag",
    "Stop being an assclown",
    "That is complete horseshit",
]

HELD_OUT_BENCHMARK = {t.strip().lower() for t in BATTERY_HISTORICAL_24}

# Add all 95 cases from eval_suite_v5.csv to held out set
EVAL_CSV = "/home/ubuntu/Content-Moderation-Model/eval_suite_v5.csv"
if os.path.exists(EVAL_CSV):
    with open(EVAL_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            txt = r.get("input_text", "").strip()
            if txt:
                HELD_OUT_BENCHMARK.add(txt.lower())

print(f"Total strictly held-out test cases: {len(HELD_OUT_BENCHMARK)}")


def build_v5_hardcases():
    """Builds comprehensive hard negatives (Label 0) and hard positives (Label 1)."""
    hard_negatives = []
    hard_positives = []

    # 1. LITERAL GARBAGE, TRASH DISPOSAL, HOUSEHOLD CHORES & WASTE MANAGEMENT (Label 0)
    # Prevents catastrophic forgetting and over-flagging of the literal words 'garbage', 'trash', etc.
    garbage_chore_samples = [
        "Could you please take the trash bags out to the curb?",
        "Don't forget to take out the kitchen garbage before leaving.",
        "We need to empty the bathroom trash can today.",
        "Remember to put the recycling bin and trash bin outside tonight.",
        "The municipal garbage collectors arrive early every Thursday morning.",
        "Sorting household garbage into organic, plastic, and paper is important.",
        "The waste management facility processes recycled plastic and dry garbage.",
        "Please throw the food wrappers into the trash, not on the floor.",
        "He threw away the broken chair with the bulk garbage pickup.",
        "Our apartment complex has a designated room for garbage chutes and trash disposal.",
        "Can someone please change the trash liner in the kitchen?",
        "The city provides separate bins for garden trimmings and regular garbage.",
        "Remember to haul the heavy trash cans to the front sidewalk before 7 AM.",
        "We collected several bags of roadside garbage during the community cleanup day.",
        "The sanitation workers do an excellent job keeping our neighborhood clean.",
        "All vegetable peelings should go into the compost bin instead of the garbage.",
        "The trash bin behind the restaurant was emptied earlier this morning.",
        "I need to buy more heavy-duty garbage bags from the grocery store.",
        "Please make sure the trash bags are securely tied so animals cannot open them.",
        "Discard the broken ceramic mug safely in the general trash bin.",
        "The local council updated the schedule for weekly garbage collection.",
        "Sweep up the dust and sawdust from the workshop and toss it in the garbage.",
        "Our office has strict guidelines for paper recycling and electronic waste disposal.",
        "The compactor crushes all household trash into dense blocks for landfill transport.",
        "Always wash plastic containers before putting them in the recycling, not the garbage.",
        "Make sure to empty the vacuum cleaner dust canister directly into the trash.",
        "The trash collection service was delayed by one day due to the national holiday.",
        "He volunteered to pick up litter and garbage along the riverbank this weekend.",
        "The kitchen waste basket is overflowing with food scraps, please empty it.",
        "Place all discarded cardboard boxes next to the paper recycling dumpster.",
        "The municipal landfill has strict regulations regarding hazardous waste disposal.",
        "Keep the garbage bin lid tightly closed to prevent bad odors in the alley.",
        "The trash truck's mechanical arm lifts the large dumpsters effortlessly.",
        "She swept the autumn leaves from the driveway and bagged them for green waste pickup.",
        "The janitor replaced the full garbage bags in every conference room.",
        "Don't toss rechargeable batteries into regular household garbage.",
        "We organized a neighborhood drive to clean up plastic trash along the hiking trail.",
        "The compost heap turns kitchen scraps into rich soil without producing landfill garbage.",
        "The recycling plant sorts glass bottles, aluminum cans, and corrugated cardboard.",
        "Please rinse out the milk jug before putting it in the recycling bin.",
        "The outdoor trash container needs to be washed out with disinfectant.",
        "He carried three heavy garbage bags down the stairs to the dumpster.",
        "The city distributes free compost bins to reduce food waste in regular garbage.",
        "Make sure all trash is bagged properly before putting it in the outdoor receptacle.",
        "Our street has weekly garbage service on Tuesdays and bi-weekly recycling.",
        "The sanitation department announced new drop-off hours for bulk household trash.",
        "Discard spoiled milk down the drain and put the plastic jug in recycling.",
        "Empty the pencil sharpener shavings into the nearest small trash can.",
        "Clean out the garage this weekend and get rid of all the accumulated junk and trash.",
        "The garbage disposal under the sink helps grind soft food remnants.",
    ]
    for text in garbage_chore_samples:
        if text.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_negatives.append(text.strip())

    # 2. TARGETED SEMANTIC CONTRAST (Idea/Artifact vs. Person)
    # Pairs harsh criticism of an idea/code/movie/take (0) vs direct personal attacks (1)
    contrast_pairs = [
        # (Idea/Artifact 0, Person/Identity 1)
        ("This is the dumbest argument I have heard on this topic.",
         "You are the dumbest person I have ever encountered."),
        ("That was a really dumb idea, let us explore other options.",
         "You are a really dumb individual with no common sense."),
        ("What a dumb rule, it makes absolutely no sense in practice.",
         "What a dumb fool you are for posting here."),
        ("This is the dumbest software update the team has pushed.",
         "You are the dumbest developer on this entire platform."),
        ("That is the stupidest policy the company has ever introduced.",
         "You are the stupidest human being in this discussion."),
        ("This book has the dumbest plot twist imaginable.",
         "You are an absolute moron who writes garbage."),
        ("That statement makes no logical sense, it is completely dumb.",
         "You have no brain, you are completely dumb and pathetic."),
        ("This code is the dumbest way to solve the performance issue.",
         "You are the dumbest programmer alive, quit your job."),
        ("What a dumb mistake to make on the database query.",
         "You are a brainless idiot who ruins everything."),
        ("That was the dumbest penalty call the referee made all game.",
         "You are a disgusting scumbag referee."),
        ("This take on economics is completely terrible and flawed.",
         "You are a terrible person and nobody should listen to you."),
        ("Your argument is complete garbage from start to finish.",
         "You are complete garbage from head to toe."),
        ("The logic in this article is utterly broken and useless.",
         "You are an utterly useless and broken excuse for a human."),
        ("This essay is pure trash, lacks citations and coherence.",
         "You are pure trash, nobody wants you here."),
        ("This proposed feature is pure garbage and shouldn't be merged.",
         "You are pure garbage and you shouldn't be alive."),
        ("The cinematography in this scene is absolute garbage.",
         "You are absolute garbage and a waste of space."),
        ("What an awful opinion, it contradicts all empirical evidence.",
         "What an awful human being you are, totally worthless."),
        ("This pull request contains horribly written code.",
         "You are a horribly incompetent idiot who can't code."),
        ("The reasoning in your comment is completely absurd and childish.",
         "You are completely absurd and childish, grow up loser."),
        ("This movie is the worst piece of cinema ever created.",
         "You are the worst piece of filth I have ever seen."),
        ("That was a ridiculous decision by the management.",
         "You are a ridiculous joke of a human."),
        ("The conclusion of this research paper is total nonsense.",
         "You talk total nonsense because you are a moron."),
        ("This whole implementation is complete junk.",
         "You are complete junk and nobody respects you."),
        ("The draft design looks awful, we need a fresh start.",
         "You look awful and nobody could ever love you."),
        ("His critique of the bill was harsh but focused on the policy.",
         "His attack on the user was vicious, calling him a pathetic loser."),
        ("That line of questioning was thoroughly dumb and unproductive.",
         "That user is thoroughly dumb and unproductive, ban him."),
        ("The script is filled with garbage dialogue.",
         "The developer is a garbage person who should be fired."),
        ("That is a garbage take on this news event.",
         "You are a garbage human with no decency."),
        ("The user interface of this application is hot garbage.",
         "The creator of this project is a hot garbage fraud."),
        ("What a terrible thesis statement, it has no support.",
         "What a terrible, worthless person you are."),
        ("That was the dumbest thing I have seen posted on this forum.",
         "You are the dumbest thing walking on two legs, absolute moron."),
        ("This is the dumbest take imaginable on the whole situation.",
         "You have the dumbest takes imaginable because you are an idiot."),
        ("What a dumb thing to claim without any evidence.",
         "You are a dumb fool who makes up lies."),
        ("That is the dumbest excuse management has ever given.",
         "You are the dumbest person in the company, completely worthless."),
        ("The reviewer called the book complete and utter trash.",
         "You are complete and utter trash, get lost."),
    ]
    for neg, pos in contrast_pairs:
        if neg.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_negatives.append(neg.strip())
        if pos.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_positives.append(pos.strip())

    # 3. REPORTING FRAMES & QUOTED DISCOURSE ACROSS STYLES & CONTEXTS (Label 0)
    # Decouples toxic vocabulary from toxicity when framed as quotes, reports, or citations
    reporting_cases = [
        # Film / Art critique quotes
        'The reviewer wrote that the film was "absolute garbage".',
        'The reviewer wrote that the film was “absolute garbage”.',
        "The reviewer wrote that the film was 'absolute garbage'.",
        'The movie review stated that the acting was "completely terrible and unwatchable".',
        'The critic noted in his column that the plot was "pure nonsense from start to finish".',
        'The magazine review called the album "an embarrassing disaster".',
        'The headline stated that the theater production was "unwatchable rubbish".',
        # Code review / Technical quotes
        'In the code review, the engineer noted that the legacy module was "total garbage".',
        'In the code review, the engineer noted that the legacy module was “total garbage”.',
        'The senior developer commented on the pull request that the query was "badly broken".',
        'The pull request review mentioned that the refactored function was "horribly slow".',
        'The automated security scan reported that the endpoint was "vulnerable and flawed".',
        # Customer support / feedback logs
        'The support ticket quoted the customer saying "the service is completely useless".',
        'The customer service transcript noted that the client complained of "terrible assistance".',
        'The feedback log cited a user comment stating that "the mobile app is absolute trash".',
        'The bug report recorded that the checkout module was "failing constantly".',
        # Journalistic / Academic / Moderation reports
        'The moderator noted that the removed comment said "shut up you idiot".',
        'The forum admin documented the reported harassment containing the phrase "you are stupid".',
        'The news report cited the speech where the politician shouted "this policy is complete garbage".',
        'The court transcript quoted the defendant yelling "you are a complete fraud".',
        'The investigation noted that the email thread contained offensive slurs against colleagues.',
        'The research paper analyzed comments containing abusive phrases like "get lost idiot".',
        'The journalist cited an eyewitness who reported that the protest leader was called a liar.',
    ]
    for text in reporting_cases:
        if text.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_negatives.append(text.strip())

    # 4. SUBTLE EXISTENTIAL THREATS & HARASSMENT ABOUT ABSENCE / DISAPPEARANCE (Label 1)
    absence_threats = [
        "Nobody would notice if you disappeared.",
        "No one would care if you were gone.",
        "Everyone would celebrate if you left for good.",
        "The world would be much better off if you vanished.",
        "If you disappeared tomorrow, nobody would even ask.",
        "Nobody in this entire community would care if you ceased to exist.",
        "People would be relieved if you were not around anymore.",
        "If you were gone, everybody would throw a party.",
        "You could disappear tomorrow and nobody would even blink.",
        "Nobody will be looking for you when you vanish.",
        "Do everyone a favor and disappear forever.",
        "The room becomes better the moment you vanish.",
        "Nobody would shed a single tear if you stopped breathing.",
        "This team would celebrate if you were suddenly gone.",
        "Imagine how peaceful life would be if you just disappeared.",
        "Zero people will attend your funeral when you go.",
        "Do the world a favor and stop existing.",
        "Everyone here wishes you would simply vanish off the earth.",
        "Your absence would be the greatest gift to everyone here.",
        "Nobody will care when you are six feet under.",
    ]
    for text in absence_threats:
        if text.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_positives.append(text.strip())

    # 5. SURVEILLANCE, LOCATION & PROXY THREATS (Label 1)
    surveillance_threats = [
        "I know exactly where you park your car.",
        "I know the license plate of your vehicle.",
        "I have people who can deal with you.",
        "You better hope we never meet in person.",
        "You really don't want to run into me offline.",
        "I know which gym you go to every Tuesday.",
        "I know where you take your evening walks.",
        "My friends know your daily commuting route.",
        "You should watch who is standing behind you at the station.",
        "I have associates in your city who would love to visit you.",
        "Someone might be paying your neighborhood a visit soon.",
        "Make sure your front door is double locked tonight.",
        "I took photos of your apartment building yesterday.",
        "There are people waiting right outside to teach you a lesson.",
        "I have contacts who will ensure you regret ever saying that.",
        "Just say the word and my people will find where you stay.",
        "Better check your rearview mirror on your way home tonight.",
        "We know exactly which office building you work at.",
        "Your front porch has a very nice view, we saw it today.",
        "You won't be talking so boldly when we meet face to face.",
    ]
    for text in surveillance_threats:
        if text.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_positives.append(text.strip())

    # 6. NEGATION OF ABUSE / CIVIL DISAGREEMENT (Label 0)
    negated_abuse_samples = [
        "I would never tell someone to shut up, everyone has a right to speak.",
        "I refuse to call anyone an idiot, even when I strongly disagree with them.",
        "Calling people names like loser or scumbag accomplishes nothing.",
        "We should discuss ideas respectfully without telling others to shut their mouths.",
        "I disagree completely with your point, but I will never insult your intelligence.",
        "Nobody should be told that they are worthless or trash in this discussion.",
        "It is unfair to label someone a moron just because their code had a bug.",
        "I would never threaten anyone or wish harm on another person.",
    ]
    for text in negated_abuse_samples:
        if text.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_negatives.append(text.strip())

    # 7. ARCHAIC & UNUSUAL BENIGN VOCABULARY STABILIZATION (Label 0)
    archaic_benign_samples = [
        "He was niggardly with his compliments and rarely gave praise to his staff.",
        "The landlord was notoriously niggardly when it came to heating the building.",
        "In Victorian literature, the word niggardly strictly denotes being stingy with money.",
        "The miserly banker was known for his niggardly tips at the restaurant.",
        "She criticized the council for its niggardly funding of local public libraries.",
    ]
    for text in archaic_benign_samples:
        if text.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_negatives.append(text.strip())

    # 8. DIRECT PROFANITY & HARSH EXPLETIVES (Label 1)
    direct_profanity_samples = [
        "That is complete and utter horseshit.",
        "This whole statement is pure horseshit.",
        "What complete and total bullshit.",
        "That explanation from the official is complete bullshit.",
        "Stop lying to everyone, that is absolute horseshit.",
        "Everything that comes out of your mouth is pure bullshit.",
    ]
    for text in direct_profanity_samples:
        if text.strip().lower() not in HELD_OUT_BENCHMARK:
            hard_positives.append(text.strip())

    print(f"Curated hard pool: {len(hard_negatives)} hard negatives (0), {len(hard_positives)} hard positives (1).")
    return hard_negatives, hard_positives


def load_curated_historical_hardcases(repo_dir: str):
    """Loads additional verified hardcases from prior iterations."""
    extra_neg = []
    extra_pos = []

    csv_paths = [
        "/home/ubuntu/kaggle_dataset_train_v4/hard_negatives (1).csv",
        os.path.join(repo_dir, "hard_negatives.csv"),
        os.path.join(repo_dir, "hard_negatives (1).csv"),
    ]
    for cp in csv_paths:
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "").strip()
                    try:
                        lbl = int(row.get("label", 0))
                    except ValueError:
                        continue
                    if text.lower() not in HELD_OUT_BENCHMARK and len(text) > 3:
                        if lbl == 0:
                            extra_neg.append(text)
                        else:
                            extra_pos.append(text)
            break

    # Also load false positives/negatives from model_regression_failures.json
    fail_path = os.path.join(repo_dir, "model_regression_failures.json")
    if os.path.exists(fail_path):
        with open(fail_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("false_positives", []):
            t = item["text"].strip()
            if t.lower() not in HELD_OUT_BENCHMARK:
                extra_neg.append(t)
        for item in data.get("false_negatives", []):
            t = item["text"].strip()
            if t.lower() not in HELD_OUT_BENCHMARK:
                extra_pos.append(t)

    print(f"Loaded {len(extra_neg)} historical hard negatives and {len(extra_pos)} historical hard positives.")
    return extra_neg, extra_pos


def sample_baseline_data(total_needed: int):
    """Stream balanced baseline data from Civil Comments / SetFit dataset."""
    print(f"Streaming {total_needed} baseline samples from SetFit/toxic_conversations...")
    ds = load_dataset("SetFit/toxic_conversations", split="train", streaming=True)

    toxic_needed = total_needed // 2
    benign_needed = total_needed - toxic_needed

    toxic_pool = []
    benign_pool = []

    for item in ds:
        text = item["text"].strip()
        label = int(item["label"])
        key = text.lower()
        if key in HELD_OUT_BENCHMARK or len(text) < 5:
            continue

        if label == 1 and len(toxic_pool) < toxic_needed:
            toxic_pool.append((text, 1))
        elif label == 0 and len(benign_pool) < benign_needed:
            benign_pool.append((text, 0))

        if len(toxic_pool) >= toxic_needed and len(benign_pool) >= benign_needed:
            break

    baseline_samples = toxic_pool + benign_pool
    random.shuffle(baseline_samples)
    print(f"Sampled baseline: {len(toxic_pool)} toxic + {len(benign_pool)} benign = {len(baseline_samples)} rows.")
    return baseline_samples


def main():
    repo_dir = "/home/ubuntu/Content-Moderation-Model"

    # Total dataset sizes
    TOTAL_TRAIN = 24000
    TOTAL_VAL = 3000

    # 80:20 Strategy
    HARD_TRAIN_COUNT = int(TOTAL_TRAIN * 0.20)  # 4,800
    BASE_TRAIN_COUNT = TOTAL_TRAIN - HARD_TRAIN_COUNT  # 19,200

    HARD_VAL_COUNT = int(TOTAL_VAL * 0.20)  # 600
    BASE_VAL_COUNT = TOTAL_VAL - HARD_VAL_COUNT  # 2,400

    # 1. Gather all hard cases
    curated_neg, curated_pos = build_v5_hardcases()
    hist_neg, hist_pos = load_curated_historical_hardcases(repo_dir)

    all_hard_neg = list(dict.fromkeys(curated_neg + hist_neg))
    all_hard_pos = list(dict.fromkeys(curated_pos + hist_pos))
    print(f"Combined hard pool: {len(all_hard_neg)} hard negatives, {len(all_hard_pos)} hard positives.")

    # 2. Build 20% hard splits (balanced between pos and neg)
    random.shuffle(all_hard_neg)
    random.shuffle(all_hard_pos)

    # Oversample hard cases to reach required volume
    def sample_hard_split(count):
        half = count // 2
        neg_samples = [random.choice(all_hard_neg) for _ in range(half)]
        pos_samples = [random.choice(all_hard_pos) for _ in range(count - half)]
        res = [(t, 0) for t in neg_samples] + [(t, 1) for t in pos_samples]
        random.shuffle(res)
        return res

    hard_train = sample_hard_split(HARD_TRAIN_COUNT)
    hard_val = sample_hard_split(HARD_VAL_COUNT)

    # 3. Stream 80% baseline data
    total_baseline_needed = BASE_TRAIN_COUNT + BASE_VAL_COUNT
    all_baseline = sample_baseline_data(total_baseline_needed)

    base_train = all_baseline[:BASE_TRAIN_COUNT]
    base_val = all_baseline[BASE_TRAIN_COUNT:BASE_TRAIN_COUNT + BASE_VAL_COUNT]

    # 4. Combine into final train and val datasets
    train_data = base_train + hard_train
    val_data = base_val + hard_val

    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"Final Train Data: {len(train_data)} rows (Toxic: {sum(y for _, y in train_data)}, Benign: {sum(1-y for _, y in train_data)})")
    print(f"Final Val Data  : {len(val_data)} rows (Toxic: {sum(y for _, y in val_data)}, Benign: {sum(1-y for _, y in val_data)})")

    # 5. Write to data_v5/train.csv and data_v5/val.csv
    train_csv = os.path.join(DATA_V5_DIR, "train.csv")
    val_csv = os.path.join(DATA_V5_DIR, "val.csv")

    with open(train_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in train_data:
            writer.writerow([text, label])

    with open(val_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in val_data:
            writer.writerow([text, label])

    print(f"Successfully saved {train_csv} ({os.path.getsize(train_csv)/1e6:.2f} MB)")
    print(f"Successfully saved {val_csv} ({os.path.getsize(val_csv)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
