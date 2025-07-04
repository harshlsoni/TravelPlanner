# 🧭 AI Travel Planner (WIP)

An intelligent agent-based travel planner that understands natural language queries like:

> *"Plan a trip from Pune to Beawar on 15 August in AC class, preferably by train or bus."*

It parses the query, fetches real-time travel options, filters results using reasoning, and recommends the best route. The project will soon be accessible via WhatsApp as a conversational bot.

---

## 🚧 Status

> **Project Status:** `In Progress` 🚀  
Agent tuning and multi-hop route reasoning are actively being refined.  
WhatsApp integration is the next deployment step.

---

## 🔍 Features (Planned / Partial)

- ✅ Natural language travel intent extraction (source, destination, date, preferences)
- ✅ Travel data scraping from platforms like Ixigo, RedBus
- ✅ Route filtering using reasoning (ReAct / LangGraph agents)
- ⏳ WhatsApp bot interface for direct user interaction
- ⏳ Deployment-ready agent orchestration using LangChain / LangGraph

---

## 🧠 Tech Stack

- **LLM & Agents**: LangChain, LangGraph, ReAct-style agents
- **Scraping**: Playwright, BeautifulSoup
- **Backend**: Python, FastAPI (planned)
- **Deployment Target**: WhatsApp (via Twilio or WhatsApp Cloud API)

---