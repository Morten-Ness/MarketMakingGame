# Quant Market-Making 5-Dice Simulator

Local interview-practice simulator for making markets on the sum of five six-sided dice.

## Running

```bash
./.venv/bin/python Core.py
```

You can also run the dice-sum game directly from its game folder:

```bash
./.venv/bin/python games/dice_sum_market/run.py
```

If dependencies are missing in a fresh environment, install them with:

```bash
pip install -r requirements.txt
```

The app reads local settings from `.env`.

At startup, the exchange speaks the game setup over text-to-speech: turn count,
first-trade ending rule, pass/tightening rule, your turn-order position, the full
turn order, whether private signals must be shared, underlying, and your private
die. The spoken setup does not include the range or expected value.

Useful game parameters in `.env`:

```env
MAX_TURNS=9
END_ON_TRADE=false
BOT_COUNT=2
RANDOMIZE_TURN_ORDER=true
ALLOW_PASS=true
MIN_TIGHTEN_INCREMENT=0.5
```

`BOT_COUNT` is the number of bot opponents. If total participants exceed the
number of dice, every die is assigned to at least one participant and the extra
participants randomly share an existing die signal. The exchange announces that
sharing is required, but does not reveal who shares with whom.

If `ALLOW_PASS=false`, participants cannot pass. They must hit, lift, or make a
market that tightens the current best market by at least `MIN_TIGHTEN_INCREMENT`
on at least one side.

## Bot Information

Before every AI turn, the exchange sends the bot:

- its private die
- game configuration: turn count, current turn, ending rule, participants, participant count, turn order, and the bot's own turn-order position
- underlying configuration: dice count, die sides, range, and unconditional expected value
- information structure: the bot's own private die value, whether private-signal sharing is required, and how many dice remain unobserved
- trading rules: whether passing is allowed and the minimum tightening increment
- the current public order book
- the current best bid and offer
- its own position and cash
- the full chronological `action_tape`
- the structured list of executed trades

These fields are generated from the same engine state that drives the user briefing and game loop, so they update automatically if the game parameters change.

The `action_tape` is ordered from the start of the game and includes public quotes, passes, hit/lift attempts, executed trades, and rejected actions.

## Voice Input

Voice input is not looking for one exact sentence. It records your speech, transcribes it with faster-whisper, then sends the transcribed text through the same command parser used for typed input.

Because the parser is intentionally small, short and explicit commands work best.

## Supported Trading Commands

### Make Or Update A Market

Use a bid, then an offer.

Good voice examples:

```text
16 at 18
I make 16 at 18
17.5 at 19
bid 16 offer 18
bid 16 ask 18
16 bid 18 offer
```

Also accepted between prices:

```text
16 @ 18
16 / 18
16 by 18
16 x 18
```

The bid must be lower than the offer, and the market must stay within the dice-sum range of `5` to `30`.

If another participant already has a market, your new market must be inside or equal to the current best market. For example, if the best market is:

```text
16 at 18
```

These are accepted as tighter or equal markets:

```text
16 at 18
16.5 at 18
16 at 17.5
16.5 at 17.5
```

These are rejected because they extend one side or widen the market without trading:

```text
17 at 19
15 at 17
15 at 19
```

These trigger trades because they cross the existing market:

```text
18 at 20
14 at 16
```

### Pass

Good voice examples:

```text
pass
skip
no market
nothing
stand aside
```

Pass only works when `ALLOW_PASS=true`.

When `ALLOW_PASS=false`, a non-crossing quote against an existing market must
tighten at least one side by `MIN_TIGHTEN_INCREMENT`. With the default increment
of `0.5`, if the current best market is:

```text
16 at 18
```

These are accepted:

```text
16.5 at 18
16 at 17.5
16.5 at 17.5
```

These are rejected:

```text
16 at 18
16.25 at 18
16 at 17.75
```

### Hit A Bid

You sell to the best available external bid.

Good voice examples:

```text
hit
hit the bid
hit the 16 bid
sell bid
selling to the bid
```

If you include a price, it must match an available bid.

### Lift An Offer

You buy from the best available external offer.

Good voice examples:

```text
lift
lift the offer
lift the 18 offer
buy offer
buy the offer
take the offer
```

If you include a price, it must match an available offer.

## Voice Settings

Useful `.env` settings:

```env
ENABLE_VOICE_INPUT=true
ENABLE_AUDIO_CUES=true
AUDIO_CUE_VOLUME=0.2
WHISPER_MODEL=tiny.en
VAD_SILENCE_MS=500
VOICE_MAX_SECONDS=30
```

Audio cues are short tones:

- your turn started
- your command was accepted by the engine
- your command was rejected or could not be parsed

If recording stops too quickly after a pause, increase:

```env
VAD_SILENCE_MS=800
```

If you need a longer maximum speaking window, increase:

```env
VOICE_MAX_SECONDS=45
```

## Bot Brain Status

At startup, the app prints whether bots are using Gemini or the local heuristic fallback:

```text
Bot brain: Gemini API (gemini-2.5-pro, temperature=0.2).
```

or:

```text
Bot brain: local heuristic fallback (programmatic).
```

## Logs

Each game keeps its logs inside its own game folder. For the dice-sum game,
scratchpads and summaries append to:

```text
games/dice_sum_market/logs/scratchpads.jsonl
games/dice_sum_market/logs/game_summaries.jsonl
```

Summary rows include `turns_used`, `bot_count`, and `user_final_pnl`. Set
`SCRATCHPAD_LOG_PATH` or `GAME_SUMMARY_LOG_PATH` in `.env` if you want a custom
location.
