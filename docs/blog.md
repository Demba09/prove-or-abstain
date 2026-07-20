# Comment j'ai construit un agent qui refuse d'agir — et pourquoi c'est le vrai hard tech

> *"Monday, 9am. Conversion is down 3.2%. The PM drops a message in Slack: 'What happened? Campaign? Bug? New users behaving differently?' The data team spends 4 hours slicing dashboards. They check segment by segment, device by device. Eventually — maybe — they find a culprit. Or maybe the drop is systemic and they're chasing noise."*

Ce scénario, tous ceux qui ont bossé dans une boîte avec un dashboard l'ont vécu. Et en 2026, la réponse qu'on nous vend partout c'est : « mets un agent IA qui analyse tes métriques automatiquement. »

**Le problème ?** Ces agents mentent. Pointés sur une métrique qui baisse, ils trouveront *toujours* une cause — même quand les données ne contiennent aucun signal exploitable. Un diagnostic faux mais formulé avec assurance est *pire* que pas de diagnostic du tout, surtout si l'agent est autorisé à agir (couper une campagne, modifier un budget, envoyer une notif).

J'ai donc construit l'inverse : **un agent dont l'autonomie est conditionnée par une preuve mathématique**, et dont la seule alternative à agir est un refus propre, motivé, nommant exactement *pourquoi* il ne peut pas localiser de cause. Pas un "je sais pas" flou — une raison précise et vérifiable.

Ce n'est pas juste une intuition. C'est un pipeline à 4 portes déterministes, un benchmark de 30 scénarios à 100% d'accuracy, et 0 faux positifs (false-ASSERT). Voici comment ça marche.

---

## Le principe : ASSERT ou ABSTAIN, pas « peut-être »

L'agent `prove-or-abstain` prend en entrée deux snapshots d'une métrique business (baseline vs current, segmentés par dimensions — device, segment, pays, plan...) et retourne l'un des deux verdicts :

| Verdict | Signification | Exemple |
|---------|--------------|---------|
| **ASSERT** | Cause trouvée et prouvée statistiquement | *"Le segment 'paid' s'est effondré (p < 0.001). Recommandation : suspendre la campagne paid."* |
| **ABSTAIN** | Aucune cause unique isolable — systémique ou diffus | *"La baisse est réelle mais uniformément répartie sur tous les segments. Escalader à un humain."* |

Le verdict ABSTAIN n'est pas un échec. C'est **la fonction de sécurité** de l'agent. Un agent autorisé à agir doit être capable de refuser d'agir quand les preuves sont insuffisantes.

Et ce n'est pas un prompt qu'on a demandé gentiment au LLM. C'est un garde-fou codé en dur dans un pipeline à 4 portes.

---

## Les 4 portes qui décident

L'agent ne « réfléchit » pas à si une cause est valide. Il teste chaque dimension candidate (device, segment, pays...) à travers 4 checkpoints déterministes :

| Porte | Condition | Ce qu'elle empêche |
|-------|-----------|-------------------|
| **Material** | \|ΔR\|/R₀ ≥ 2% | Ignorer une variation négligeable |
| **Localized** | Concentration ≥ 0.55 | Blâmer un segment alors que la baisse est diffuse |
| **Significant** | Z-test p ≤ 0.01 (ou n ≥ 1000 pour les métriques somme) | Agir sur du bruit d'échantillonnage |
| **Clean** | Interaction ≤ 0.50 | Confondre un effet de composition avec un effet de taux |

Si les 4 passent → ASSERT. Si une seule échoue → ABSTAIN, avec la raison exacte.

**Exemple concret :** Même baisse de 3.2% sur le taux de conversion. Scénario 1 : seul le segment paid a chuté → ASSERT segment=paid. Scénario 2 : tous les segments ont baissé de la même quantité → ABSTAIN (cause diffuse, concentration = 0.25 < 0.55). Même ampleur, verdicts opposés — parce que le monde est différent.

Le test de significativité est un vrai two-proportion z-test, pas un seuil magique : une baisse parfaitement concentrée sur 6 000 utilisateurs passe (p < 1e-5), la même baisse sur 60 utilisateurs échoue (p = 0.55). C'est ça qui distingue un signal d'un artefact.

---

## Où le LLM intervient (et où il n'intervient pas)

Qwen (via DashScope) fait exactement 4 choses :

1. **Ordonner** les dimensions candidates (par où commencer à chercher ?)
2. **Phraser** le rapport final
3. **Router** les questions en langage naturel vers le bon scénario
4. **Mapper** les colonnes d'une source inconnue vers le format attendu

Partout ailleurs, **c'est pandas et numpy qui décident**. La décomposition mathématique est exacte (résidu zéro, vérifiée contre un oracle indépendant). Le LLM ne calcule jamais un nombre. Un mode mock (`QWEN_MOCK=1`) le remplace par des templates déterministes — et le verdict est **bit-identique** avec ou sans LLM.

C'est le contrat de sécurité : Qwen conduit le chemin, les gates décident du verdict.

---

## Architecture : graph ET agent loop

Le projet a deux moteurs d'investigation interchangeables, produisant des verdicts identiques (prouvé par le benchmark) :

**Mode graph** (`graph.py` + `nodes.py`) : un StateGraph LangGraph à 7 nœuds (`detector → hypothesizer → investigator → verifier → driller → actuator → reporter`) avec une boucle conditionnelle bornée à `len(dims)` itérations. Déterministe, traçable.

**Mode agent** (`agent_loop.py`) : Qwen orchestre l'investigation via des appels d'outils (`test_dimension`, `drill`, `finalize`). Plus flexible, mais un guard de déterminisme (`_finalize_verdict`) empêche un Qwen paresseux ou erratique de produire un faux ABSTAIN.

Les deux modes appellent les mêmes fonctions mathématiques. Le choix du mode change la surface d'orchestration, pas le résultat.

---

## Un benchmark qui dit la vérité (30 scénarios, 100% accuracy)

Le benchmark contient 30 scénarios synthétiques avec une ground truth connue à l'avance (on sait quel scénario a été « planté » avec quel bug). Catégories :

- **7 clean → ASSERT** (collapse d'un segment ou device)
- **3 clean → ABSTAIN** (collapse réparti sur 2+ segments)
- **5 diffuse → ABSTAIN** (baisse uniforme)
- **3 mixshift → ABSTAIN** (composition + taux bougent ensemble)
- **3 deep → ASSERT + drill-down** (un seul croisement segment×device)
- **3 edge cases** (métrique somme, tiny sample, single dimension)
- **3 noisy** (baisse réelle avec jitter)

Résultat : **30/30 en mode graph ET agent — 100% accuracy, 0% false-ASSERT, ECE 0.19**. Le benchmark est exécuté en CI et le pipeline refuse de passer si l'accuracy tombe sous 100%.

---

## Agent-ready : le code a été pensé pour les agents IA

Un aspect qui me tient à cœur : le projet est documenté pour les agents, pas juste pour les humains.

- **`AGENTS.md`** : règles non-négociables, invariants, commandes, conventions, pièges à éviter
- **`project-context.md`** : architecture, décisions de design, glossaire, flux de données
- **105 tests** automatisés, ruff à 0 erreurs, CI en matrix 3.12 + 3.13
- **Thread-safety** : tous les globals partagés (singleton LLM, compteurs, rate limiter, accès SQLite) sont lock-guardés
- **Rate limiting** : 60 req/min/IP, sliding window
- **Docker** : non-root user, HEALTHCHECK, port unique
- **`pyproject.toml`** : le projet est pip-installable (`pip install -e .`)

---

## Essayez-le en 2 secondes

```bash
git clone https://github.com/Demba09/prove-or-abstain
cd prove-or-abstain
pip install -r requirements.txt
QWEN_MOCK=1 uvicorn api.app:app
# Puis ouvrez http://localhost:8000
```

Ou via Docker :
```bash
docker build -t prove-or-abstain .
docker run --rm -p 8000:8000 -e QWEN_MOCK=1 prove-or-abstain
```

Le mode mock ne nécessite aucune clé API — tout tourne en local, déterministe, et donne les mêmes verdicts qu'avec un vrai Qwen.

---

## Ce qui est vraiment dur (et ce qui est facile)

Ce qui est **facile** : demander à GPT/O1/Claude « analyse cette baisse de métrique et dis-moi pourquoi ». Le LLM va produire un texte plausible. Il va nommer une cause. Il aura l'air confiant. Et il aura raison... ou tort. Vous ne saurez pas.

Ce qui est **dur** : construire un système où la frontière entre « je sais » et « je ne sais pas » est mathématiquement vérifiable. Où l'ABSTAIN n'est pas une option mais une **propriété de sûreté**. Où vous pouvez prouver, reproductiblement, que le système ne ment pas.

C'est ça que `prove-or-abstain` résout. Et c'est open source.

---

**Repo** : [github.com/Demba09/prove-or-abstain](https://github.com/Demba09/prove-or-abstain)
**Built with** : Python · FastAPI · LangGraph · Qwen (DashScope) · pandas · numpy · Docker
**Licence** : MIT
