"""
Грэм-анализ кредитоспособности эмитента для облигаций.
Принимает словарь с финансовыми показателями (в тыс. руб),
возвращает рейтинг, баллы и вердикт.
"""
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
import yaml
import numpy as np
from dataclasses import dataclass, field

# ------------------ Data Structures ------------------
@dataclass
class CreditReport:
    total_score: float          # 0-100
    rating: str                 # A, B, C, D
    metrics: Dict[str, float]   # рассчитанные значения метрик
    passed: Dict[str, bool]     # пройдена ли каждая метрика
    risks: List[str]            # список непройденных метрик
    verdict: str                # итоговая рекомендация по облигациям


# ------------------ PDF Extraction (оставлен для совместимости) ------------------
def extract_financials_with_llm(
    pdf_path: str,
    llm_callable: Callable[[str, str], str]
) -> dict:
    """
    Отправляет PDF в LLM и получает словарь с основными фин. показателями.
    (не используется в текущей версии, оставлен для обратной совместимости)
    """
    with open(pdf_path, "rb") as f:
        b64_pdf = base64.b64encode(f.read()).decode()

    prompt = (
        "Извлеки из годовой финансовой отчётности (РСБУ или МСФО) следующие показатели в тыс. рублей, "
        "без пробелов и единиц измерения. Если какого-то показателя нет, поставь null.\n"
        "Показатели:\n"
        "- Выручка\n"
        "- EBITDA (если нет, рассчитай как прибыль от продаж + амортизация)\n"
        "- EBIT (прибыль до вычета процентов и налогов)\n"
        "- Чистая прибыль\n"
        "- Проценты к уплате\n"
        "- Денежный поток от операционной деятельности (CFO)\n"
        "- Оборотные активы\n"
        "- Краткосрочные обязательства\n"
        "- Денежные средства и эквиваленты\n"
        "- Дебиторская задолженность (краткосрочная)\n"
        "- Собственный капитал\n"
        "- Всего активов\n"
        "- Долгосрочный долг (займы и кредиты)\n"
        "- Краткосрочный долг (займы и кредиты)\n"
        "- Выплаченные дивиденды (если есть)\n"
        "Верни ТОЛЬКО валидный JSON-словарь с этими ключами и числовыми значениями."
    )

    response = llm_callable(prompt, b64_pdf)
    data = json.loads(response)
    for k in data:
        if data[k] is None:
            data[k] = 0.0
    return data


# ------------------ Graham Analyzer (обновлённый) ------------------
class CreditAnalyzer:
    def __init__(self, config_dir: str = "config"):
        """
        Загружает все конфиги из папки config_dir.
        Ожидает файлы вида creditworthiness_criteria_<sector>.yaml
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        
        # Загружаем все YAML-файлы, начинающиеся с creditworthiness_criteria_
        for f in self.config_dir.glob("creditworthiness_criteria_*.yaml"):
            sector = f.stem.replace("creditworthiness_criteria_", "")
            with open(f, encoding='utf-8') as file:
                self.configs[sector] = yaml.safe_load(file)
        
        # Если нет ни одного конфига — выбросить ошибку
        if not self.configs:
            raise FileNotFoundError(
                f"Не найдено ни одного конфига в {config_dir}. "
                "Ожидаются файлы creditworthiness_criteria_*.yaml"
            )
        
        # Дефолтный конфиг — industrial (если есть)
        self.default_sector = 'industrial' if 'industrial' in self.configs else list(self.configs.keys())[0]

    def analyze(self, financials: dict, sector: str = 'industrial') -> CreditReport:
        """
        Анализирует финансовые показатели и возвращает CreditReport.
        financials – словарь с русскими ключами (в тыс. руб).
        sector – 'industrial' или 'bank' (или другой, если есть конфиг).
        """
        # Выбираем конфиг по сектору
        cfg = self.configs.get(sector, self.configs.get(self.default_sector))
        if cfg is None:
            raise ValueError(f"Нет конфига для сектора {sector}")

        m = {}
        # 1. Текущая ликвидность
        ca = financials.get('Оборотные активы', 0)
        cl = financials.get('Краткосрочные обязательства', 1)
        m['current_ratio'] = ca / cl if cl else float('inf')

        # 2. Быстрая ликвидность
        cash = financials.get('Денежные средства и эквиваленты', 0)
        recv = financials.get('Дебиторская задолженность', 0)
        quick_assets = cash + recv
        m['quick_ratio'] = quick_assets / cl if cl else float('inf')

        # 3. Коэффициент автономии
        equity = financials.get('Собственный капитал', 0)
        total_assets = financials.get('Всего активов', 1)
        m['equity_ratio'] = equity / total_assets if total_assets else 0

        # 4. Долг / Собственный капитал
        short_debt = financials.get('Краткосрочный долг', 0)
        long_debt = financials.get('Долгосрочный долг', 0)
        total_debt = short_debt + long_debt
        m['debt_to_equity'] = total_debt / equity if equity != 0 else float('inf')

        # 5. Долг / EBITDA
        ebitda = financials.get('EBITDA', 0)
        m['debt_to_ebitda'] = total_debt / ebitda if ebitda > 0 else float('inf')

        # 6. Покрытие процентов (ICR)
        ebit = financials.get('EBIT', 0)
        interest = financials.get('Проценты к уплате', 0)
        m['icr'] = ebit / interest if interest != 0 else float('inf')

        # 7. ROA
        net_income = financials.get('Чистая прибыль', 0)
        m['roa'] = net_income / total_assets if total_assets else 0

        # 8. CFO / краткосрочный долг
        cfo = financials.get('Денежный поток от операционной деятельности', 0)
        m['cfo_to_short_debt'] = cfo / short_debt if short_debt != 0 else float('inf')

        # 9. ROE (добавлено для банков)
        m['roe'] = net_income / equity if equity != 0 else 0

        # Оценка по шкале из конфига
        score = 0.0
        max_score = sum(v['weight'] for v in cfg['metrics'].values())
        passed = {}
        risks = []

        for metric_name, rule in cfg['metrics'].items():
            val = m.get(metric_name)
            if val is None or np.isinf(val) or np.isnan(val):
                passed[metric_name] = False
                risks.append(metric_name)
                continue

            condition = rule['condition']
            threshold = rule['threshold']
            weight = rule['weight']

            if condition == '>':
                ok = val > threshold
            elif condition == '<':
                ok = val < threshold
            else:
                ok = False

            passed[metric_name] = ok
            if ok:
                score += weight
            else:
                # Частичный балл, если положительное значение
                if val > 0 and threshold > 0:
                    partial = min(val / threshold, 1.0) * weight
                    score += partial
                risks.append(metric_name)

        normalized = (score / max_score) * 100 if max_score > 0 else 0

        # Рейтинг
        if normalized >= 80:
            rating = 'A'
        elif normalized >= 60:
            rating = 'B'
        elif normalized >= 40:
            rating = 'C'
        else:
            rating = 'D'

        # Вердикт по облигациям
        if rating in ('A', 'B'):
            verdict = "Эмитент способен в полном объёме обслуживать долг. Низкий кредитный риск."
        elif rating == 'C':
            verdict = (
                f"Приемлемый уровень риска. Требуется мониторинг. "
                f"Слабые метрики: {', '.join(risks[:3])}."
            )
        else:
            verdict = (
                f"Высокий риск дефолта. Не рекомендуется для облигационных стратегий. "
                f"Критичные проблемы: {', '.join(risks)}."
            )

        return CreditReport(
            total_score=normalized,
            rating=rating,
            metrics=m,
            passed=passed,
            risks=risks,
            verdict=verdict
        )