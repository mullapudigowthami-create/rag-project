# ============================================
# Project 4: Sentiment Analysis using AI
# ============================================

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================
# Step 1: Create the Analyzer
# ============================================
analyzer = SentimentIntensityAnalyzer()

# ============================================
# Step 2: Sample Reviews to Analyze
# ============================================
reviews = [
    "This phone is absolutely amazing! Best purchase ever 😍",
    "Terrible product, broke after 2 days. Very disappointed.",
    "The delivery was okay, nothing special.",
    "I LOVE this! Highly recommend to everyone!!",
    "Not bad, but could be better.",
    "Worst experience of my life. Never buying again! 😡",
    "Pretty good quality for the price.",
    "It's fine, does the job.",
    "Absolutely fantastic! Exceeded all my expectations 🎉",
    "Horrible customer service. Very rude staff.",
]

# ============================================
# Step 3: Analyze Each Review
# ============================================
print("=" * 55)
print("        Sentiment Analysis using AI (VADER)")
print("=" * 55)

results = []
for i, review in enumerate(reviews, 1):
    score = analyzer.polarity_scores(review)
    compound = score['compound']

    # Classify sentiment
    if compound >= 0.05:
        sentiment = "POSITIVE 😊"
        color = "green"
    elif compound <= -0.05:
        sentiment = "NEGATIVE 😠"
        color = "red"
    else:
        sentiment = "NEUTRAL  😐"
        color = "gray"

    results.append({
        "review": review,
        "compound": compound,
        "sentiment": sentiment,
        "color": color
    })

    print(f"\nReview {i}: {review}")
    print(f"  → Score: {compound:.2f}  |  Sentiment: {sentiment}")

# ============================================
# Step 4: Let User Type Their Own Review
# ============================================
print("\n" + "=" * 55)
print("        Try Your Own Review!")
print("=" * 55)

while True:
    user_input = input("\nEnter a review (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        print("Exiting... Bye! 👋")
        break

    score = analyzer.polarity_scores(user_input)
    compound = score['compound']

    if compound >= 0.05:
        sentiment = "POSITIVE 😊"
    elif compound <= -0.05:
        sentiment = "NEGATIVE 😠"
    else:
        sentiment = "NEUTRAL  😐"

    print(f"  → Score: {compound:.2f}  |  Sentiment: {sentiment}")
    print(f"  → Breakdown: Positive={score['pos']:.2f} | "
          f"Negative={score['neg']:.2f} | Neutral={score['neu']:.2f}")

# ============================================
# Step 5: Visualize Results
# ============================================
compounds  = [r["compound"] for r in results]
colors     = [r["color"] for r in results]
labels     = [f"Review {i+1}" for i in range(len(results))]

plt.figure(figsize=(12, 6))

bars = plt.bar(labels, compounds, color=colors, edgecolor='black', width=0.6)

# Add score labels on bars
for bar, val in zip(bars, compounds):
    ypos = val + 0.02 if val >= 0 else val - 0.05
    plt.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:.2f}", ha='center', fontsize=9, fontweight='bold')

plt.axhline(y=0.05,  color='green', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=-0.05, color='red',   linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=0,     color='black', linewidth=0.8)

plt.ylim(-1.2, 1.2)
plt.xlabel("Reviews", fontsize=12)
plt.ylabel("Sentiment Score (-1 = Negative, +1 = Positive)", fontsize=11)
plt.title("Sentiment Analysis of Reviews using AI", fontsize=14, fontweight='bold')
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)

# Legend
positive_patch = mpatches.Patch(color='green', label='Positive')
negative_patch = mpatches.Patch(color='red',   label='Negative')
neutral_patch  = mpatches.Patch(color='gray',  label='Neutral')
plt.legend(handles=[positive_patch, negative_patch, neutral_patch])

plt.tight_layout()
plt.show()