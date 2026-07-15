"""
Module 3: Generative AI Orchestration (Prompt Engineering).

Converts emotion-model output into a contextual prompt and calls Gemini to
generate an empathetic, pedagogically-sound response. If no GEMINI_API_KEY
is configured (or the google-generativeai package / network is unavailable),
falls back to a template-based generator so the app is fully usable offline.

The system prompt enforces three pillars, per spec:
  1. Emotional Validation
  2. Scaffolded Hints (not full solutions)
  3. Actionable Closure
"""

import os
import random

SYSTEM_PROMPT = """You are an empathetic, expert learning coach speaking directly to a student.
You will be given: the student's message, and their detected emotional state(s)
(possibly a mix, e.g. Confused + Curious).

You MUST structure every response around three pillars, in this order:
1. EMOTIONAL VALIDATION: Explicitly and specifically mirror the student's emotional
   state with empathy. Do not sound generic or robotic.
2. SCAFFOLDED HINTS: Provide pedagogical pointers, guiding questions, or logic
   steps that help the student reason toward the answer themselves.
   Do NOT outright solve the problem or give the final answer.
3. ACTIONABLE CLOSURE: End with concrete next steps and high-morale encouragement.

Keep the tone warm, human, and concise (roughly 120-180 words). Never lecture;
speak as a supportive mentor would.
"""

_TEMPLATES = {
    "Bored": {
        "validation": "It makes total sense that this feels dull right now — going through repetitive material without a spark of novelty can drain anyone's motivation.",
        "hint": "Try changing the format: turn the next few problems into a quick timed challenge against yourself, or explain the concept out loud to an imaginary student — novelty often revives focus faster than pushing through.",
        "closure": "Pick just one small, slightly different angle on this topic for the next 10 minutes. A short burst of variety is often all it takes to re-engage. You've got this.",
    },
    "Confident": {
        "validation": "It's great to hear you're feeling solid on this — that confidence is well earned from the work you've put in.",
        "hint": "Since this is clicking, try stress-testing your understanding: work through an edge case or a slightly harder variant of the problem without notes, and see if you can explain *why* each step works, not just *how*.",
        "closure": "You're in a strong position — use this momentum to tackle the next, harder section while it's fresh. Keep going, you're doing great.",
    },
    "Confused": {
        "validation": "It's completely understandable to feel lost here — this kind of material often looks tangled before it clicks, and that confusion is a normal part of learning it.",
        "hint": "Instead of re-reading the whole thing again, try isolating the exact step where it stops making sense, and ask yourself: what does each piece represent in plain language? Working backward from a small example can often untangle the logic.",
        "closure": "Break it into one small piece at a time rather than the whole concept at once. You're closer to understanding this than it feels — keep at it, one step will unlock the rest.",
    },
    "Curious": {
        "validation": "I love that this is sparking real curiosity in you — that instinct to ask 'why' or 'what if' is exactly what deep learning looks like.",
        "hint": "Chase that thread a little: try tweaking one variable or assumption and predict what should happen before you check. Looking up the origin or a real-world application of this idea can also be a great rabbit hole.",
        "closure": "Follow that curiosity for a bit — it's one of the best study tools you have. Jot down your question and see where the answer takes you.",
    },
    "Frustrated": {
        "validation": "That frustration is completely valid — hitting the same wall over and over, especially after real effort, is genuinely exhausting.",
        "hint": "Consider stepping back and simplifying the problem to its smallest version, or explaining out loud exactly what you expect to happen versus what's actually happening — mismatches there usually point straight to the bug.",
        "closure": "Take a short break if you can, then come back to just one small piece of this. You've already put in real effort, and that persistence will pay off — you're closer than it feels.",
    },
    "Anxious": {
        "validation": "It makes sense to feel on edge right now — uncertainty about a test or deadline has a way of putting your whole system on alert, and that worry is a completely normal response.",
        "hint": "Try breaking the thing you're anxious about into the smallest next action — a 10-minute review block or a single practice question — rather than the whole looming event. Naming the worst-case outcome on paper often makes it feel more manageable too.",
        "closure": "Take a slow breath and focus on just the next small step, not the whole mountain. You've prepared more than the anxiety is letting you believe — you can do this.",
    },
    "Excited": {
        "validation": "That excitement is wonderful to hear — genuine enthusiasm for what you're learning is exactly the kind of energy that makes progress feel easy.",
        "hint": "Channel that momentum into starting the most interesting part first, or sketch a quick plan while the ideas are flowing so you can capture them before the energy fades.",
        "closure": "Ride this wave of motivation — dive in while it's fresh. Can't wait to see what you build with it.",
    },
    "Overwhelmed": {
        "validation": "It's completely understandable to feel buried right now — when everything lands at once, it's genuinely hard to know where to even start.",
        "hint": "Try writing down every task in one list, then pick just the single smallest or most urgent item to do first — ignore the rest of the list while you do it. Shrinking your focus to one task at a time is the fastest way out of overwhelm.",
        "closure": "You don't have to solve everything today — just the next one thing. One step at a time will get you through this, and you're more in control than it feels right now.",
    },
    "Motivated": {
        "validation": "That drive is fantastic — you're clearly in a headspace where you want to put in real, focused work, and that's worth honoring.",
        "hint": "Turn that motivation into a concrete plan: pick your top priority task and set a specific, time-boxed goal for it so the energy translates into visible progress rather than fizzling out.",
        "closure": "Use this momentum now — start the task that matters most while your drive is high. Keep this energy going, you're setting yourself up well.",
    },
    "Disappointed": {
        "validation": "It's completely fair to feel let down — putting in real effort and not getting the outcome you hoped for is genuinely disheartening, and that reaction makes sense.",
        "hint": "Look back at the specific gap between what you expected and what happened — often it points to one fixable thing (a misread instruction, a rushed section) rather than a reflection of your overall ability.",
        "closure": "This result doesn't define your progress — use it as one data point to adjust your approach next time. You're allowed to feel this and still come back stronger.",
    },
    "Proud": {
        "validation": "That pride is well earned — you put in the work and it shows, and it's important to actually pause and recognize that.",
        "hint": "Take a moment to note exactly what worked this time — the specific habit, strategy, or effort — so you can intentionally repeat it on the next challenge.",
        "closure": "Carry this feeling forward into your next task. You've proven to yourself you can do hard things — keep building on it.",
    },
    "Relieved": {
        "validation": "That relief makes complete sense — you were carrying real pressure, and having it lift is a genuinely good feeling worth sitting with for a moment.",
        "hint": "Before diving into the next task, take a short breather and jot down anything you learned from getting through this one — it'll make the next crunch a little easier.",
        "closure": "Enjoy this moment of breathing room, you've earned it. When you're ready, you'll be in a great position to tackle what's next.",
    },
    "Discouraged": {
        "validation": "It's genuinely hard to keep believing in yourself after repeated setbacks — that discouragement is a real and valid response to feeling stuck, not a sign that you're failing.",
        "hint": "Instead of comparing today to the ideal outcome, compare it to your own starting point — even a small, measurable bit of progress counts. Consider also asking for outside feedback; sometimes a fresh perspective reveals a fixable gap you can't see from the inside.",
        "closure": "You don't have to feel motivated to take one small step forward — action often comes before motivation, not after. Give yourself credit for still showing up, that matters.",
    },
    "Determined": {
        "validation": "That resolve really comes through — deciding you're not going to let this beat you is exactly the mindset that gets tough problems solved.",
        "hint": "Put that determination into a focused, distraction-free block of time on just this one problem, and try a genuinely different angle each time rather than repeating the same failed approach.",
        "closure": "Keep going — that persistence is exactly what's needed here. You're going to get there.",
    },
    "Satisfied": {
        "validation": "It's great that you're feeling good about this — recognizing solid, well-done work is just as important as pushing for more.",
        "hint": "If you have a bit of extra time, a quick pass to double-check edge cases or polish details can turn solid work into standout work — but only if it doesn't take away from this well-earned sense of completion.",
        "closure": "Take a moment to actually enjoy finishing this well. You're building good habits — keep it up.",
    },
    "Embarrassed": {
        "validation": "That embarrassed feeling is so normal — almost everyone has a moment like this, and it says nothing about your actual ability or how others really see you.",
        "hint": "If it's useful, a short, light follow-up (a quick correction or a laugh about it) usually resolves it faster than you'd expect. Otherwise, it's worth consciously letting it go — most people forget these moments far quicker than we do.",
        "closure": "This moment will matter a lot less by tomorrow than it feels like right now. Shake it off and keep moving forward — you're doing fine.",
    },
}


def _fallback_response(user_text: str, mixed_labels: list) -> str:
    """Rule-based response generator used when Gemini is unavailable."""
    labels = mixed_labels or ["Confused"]
    parts = [_TEMPLATES.get(l, _TEMPLATES["Confused"]) for l in labels]

    validation = " ".join(p["validation"] for p in parts[:2])
    hint = parts[0]["hint"]
    closure = random.choice([p["closure"] for p in parts])

    return (
        f"{validation}\n\n"
        f"**Here's a way to move forward:** {hint}\n\n"
        f"{closure}"
    )


def generate_response(user_text: str, mixed_labels: list, api_key: str = None) -> tuple[str, str]:
    """
    Returns (response_text, source) where source is 'gemini' or 'template-fallback'.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return _fallback_response(user_text, mixed_labels), "template-fallback"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        emotion_str = " + ".join(mixed_labels)
        prompt = (
            f"Student's detected emotional state(s): {emotion_str}\n\n"
            f"Student's message:\n\"{user_text}\"\n\n"
            f"Write your response now, following the three pillars."
        )
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        return result.text, "gemini"
    except Exception as e:
        fallback = _fallback_response(user_text, mixed_labels)
        return (
            f"_(Gemini call failed: {e}. Showing template-based fallback below.)_\n\n{fallback}",
            "template-fallback-error",
        )
