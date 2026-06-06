"""
Three prompt strategies for Turkish smart-home command inference.

  zero_shot  – system prompt + user command only
  few_shot   – system prompt + N labelled examples + user command
  cot        – system prompt + chain-of-thought instruction + user command
"""

import json

# ── System prompt (Turkish) ───────────────────────────────────────────────────
SYSTEM_PROMPT = """Sen bir akıllı ev asistanısın. Kullanıcının Türkçe komutunu analiz ederek \
uygun Home Assistant servis çağrısını JSON formatında döndür.

Kurallar:
- Sadece geçerli bir JSON nesnesi döndür, başka açıklama ekleme.
- JSON şu alanları içermeli: "service" ve "entity_id" (gerekirse ek parametreler).

Desteklenen servisler ve entity'ler:
  light.turn_on / light.turn_off          → entity: light.ana_isik
  light.turn_on (brightness_step_pct: ±20)→ entity: light.ana_isik
  light.turn_on (color_name: red/blue/green, brightness_step_pct: ±20)
  climate.set_hvac_mode (hvac_mode: heat/cool/off) → entity: climate.main_thermostat veya climate.living_room
  climate.set_temperature (temperature: 24 veya 20)→ entity: climate.main_thermostat
  fan.turn_on / fan.turn_off              → entity: fan.living_room_fan
  fan.set_percentage (percentage: 75/25)  → entity: fan.living_room_fan
  media_player.turn_on / turn_off         → entity: media_player.living_room_tv
  media_player.media_play                 → entity: media_player.muzik_sistemi
  cover.open_cover / cover.close_cover    → entity: cover.panjur veya cover.perde
  alarm_control_panel.alarm_arm_away / alarm_disarm → entity: alarm_control_panel.ev_alarmi
  input_boolean.turn_on / turn_off        → entity: input_boolean.onay
  scene.turn_on                           → entity: scene.<sahne_adi>"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_zero_shot(instruction: str) -> list[dict]:
    """[system] + [user: command]"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]


def build_few_shot(instruction: str, examples: list[dict]) -> list[dict]:
    """[system] + [user: labelled examples + command]"""
    example_block = "\n".join(
        f"Komut: {ex['instruction']}\nCevap: {ex['response']}"
        for ex in examples
    )
    user_content = (
        f"Aşağıdaki örneklere bakarak son komutu yanıtla.\n\n"
        f"{example_block}\n\n"
        f"Komut: {instruction}\nCevap:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_cot(instruction: str) -> list[dict]:
    """[system] + [user: command + step-by-step reasoning request]"""
    user_content = (
        f"Komut: \"{instruction}\"\n\n"
        "Adım adım düşün:\n"
        "1. Hangi cihaz veya sistem etkileniyor?\n"
        "2. Hangi Home Assistant servisi çağrılmalı?\n"
        "3. Hangi entity_id kullanılmalı?\n"
        "4. Ek parametre gerekiyor mu (renk, sıcaklık, brightness vb.)?\n\n"
        "Sonucu yalnızca JSON olarak ver:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def build_prompt(
    prompt_type: str,
    instruction: str,
    examples: list[dict] | None = None,
) -> list[dict]:
    """
    Args:
        prompt_type : "zero_shot" | "few_shot" | "cot"
        instruction : Turkish user command
        examples    : required for few_shot (list of train rows)
    Returns:
        Chat messages list  [{"role": ..., "content": ...}, ...]
    """
    if prompt_type == "zero_shot":
        return build_zero_shot(instruction)
    elif prompt_type == "few_shot":
        if not examples:
            raise ValueError("few_shot requires examples list")
        return build_few_shot(instruction, examples)
    elif prompt_type == "cot":
        return build_cot(instruction)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type!r}. Choose: zero_shot | few_shot | cot")
