# Bittensor Subnet 90 - DegenBrain

**Status: ✅ Subnet fully operational — 4 validators and 256 neurons active**

A Bittensor subnet for automated verification of prediction market statements through distributed consensus.

## 🛠️ Prerequisites

- **OS**: Linux (Ubuntu 22.04 LTS recommended)
- **TAO Balance**: 1+ TAO for registration costs (~1 TAO per hotkey)
- **RAM**: 1GB+ per miner
- **Storage**: 10GB+ available space

---

## 🚀 Quick Start

### 1. Install System Dependencies
```bash
# Update system and install Python 3.11 + dev tools
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 python3.11-venv python3.11-dev git curl build-essential -y
```

### 2. Clone Repository
```bash
git clone https://github.com/degenpredict/bittensor-subnet-90-brain.git
cd bittensor-subnet-90-brain
```

### 3. Setup Environment
```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch CPU version first (critical - prevents startup hangs)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Install Node.js dependencies for PM2 ecosystem
npm install

# Install everything else
pip install -r requirements.txt
```

### 4. Create Wallet & Register
```bash
# Create coldkey first (your main wallet - keep this secure!)
btcli wallet new_coldkey --wallet.name my_wallet

# Create hotkey for first miner
btcli wallet new_hotkey --wallet.name my_wallet --wallet.hotkey miner_1

# Register on subnet (costs ~1 TAO each)
btcli subnets register --netuid 90 --wallet.name my_wallet --wallet.hotkey miner_1

# Optional: Create additional miners and register them
# btcli wallet new_hotkey --wallet.name my_wallet --wallet.hotkey miner_2
# btcli subnets register --netuid 90 --wallet.name my_wallet --wallet.hotkey miner_2
```

### 5. Configure
```bash
# Copy configuration template
cp .env.example .env
```

Your `.env` file will contain these default names:
```bash
WALLET_NAME=my_wallet
MINER_1_HOTKEY=miner_1
```

**Important**: If you used different wallet/hotkey names, edit with: `nano .env`

For custom hotkey names, you can also set:
```bash
MINER_1_HOTKEY=your_custom_hotkey_name
VALIDATOR_HOTKEY=your_validator_name
```

### 6. Start Everything
```bash
# Start all processes with available hotkeys
pm2 start ecosystem.config.js

# Check status
pm2 status
pm2 logs
```

### 7. Useful Commands
```bash
pm2 restart ecosystem.config.js    # Restart all
pm2 stop all                       # Stop all
pm2 logs miner_1                   # View specific logs
pm2 monit                          # Real-time monitoring
```

**See `pm2-commands-reference.md` for complete PM2 command reference.**

---

## 🔄 System Overview

The subnet enables automated verification of prediction statements by distributing verification tasks to miners who provide evidence-based resolution decisions.

### How It Works

1. **Validator** fetches statement batches from `https://api.subnet90.com/api/test/next-chunk` (every 16+ minutes due to rate limiting)
2. **Distributes** statements to registered miners on subnet 90
3. **Miners** query `https://api.subnet90.com/api/resolutions/{id}` for official resolutions (training mode)
4. **Miners** return resolution + confidence + evidence to validator
5. **Validator** calculates consensus and scores miners based on performance
6. **Sets weights** on Bittensor to reward high-performing miners

---

## ⚙️ Configuration

Miners support different verification strategies. Edit `MINER_STRATEGY` in your `.env` file:

#### Strategy Comparison:

| Strategy | API Keys Required | Description |
|----------|------------------|-------------|
| **dummy** | None | Simple mock responses - perfect for testing |
| **ai_reasoning** | AI keys required | Full independence - analyzes statements without assistance |

### LLM Provider Configuration

The `ai_reasoning` strategy supports multiple LLM providers. Configure by setting `LLM_PROVIDER` and the corresponding API key:

#### Supported Providers:

| Provider | Description | Models Available |
|----------|-------------|------------------|
| **openai** | OpenAI GPT models | gpt-4o, gpt-4-turbo, gpt-3.5-turbo |
| **anthropic** | Anthropic Claude models | claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307 |
| **groq** | Groq (LLaMA 3) | llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768 |
| **gemini** | Google Gemini | gemini-1.5-pro, gemini-pro |
| **openrouter** | OpenRouter (Mistral, etc.) | mistralai/mistral-7b-instruct + many others |
| **chutes** | Bittensor decentralized inference | Any model deployed on Chutes |

#### Configuration Examples:

**OpenAI:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
```

**Anthropic Claude:**
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

**Groq (Fast LLaMA 3):**
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
```

**Google Gemini:**
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro
```

**OpenRouter (Mistral):**
```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
```

**Chutes (Bittensor Decentralized):**
```bash
LLM_PROVIDER=chutes
CHUTES_CPK_API_KEY=your_cpk_api_key_here
CHUTES_SLUG=your-username-model-slug
CHUTES_MODEL=unsloth/Llama-3.2-3B-Instruct
```

### Hotkey Configuration

By default, the system looks for these hotkey names:
- **Miners**: `miner_1`, `miner_2`, `miner_3`
- **Validator**: `validator`

You can customize hotkey names using environment variables:

```bash
# Custom hotkey names
MINER_1_HOTKEY=my_custom_miner
MINER_2_HOTKEY=another_miner
MINER_3_HOTKEY=third_miner
VALIDATOR_HOTKEY=my_validator
```

This is useful if you want to use different naming conventions or run multiple instances.

### Additional Data Sources

#### CoinGecko API (Crypto Data):
```bash
COINGECKO_API_KEY=your_coingecko_api_key_here
```

#### Complete Example Configuration:
```bash
# Basic setup
WALLET_NAME=my_wallet
HOTKEY_NAME=miner_1
API_URL="https://api.subnet90.com"

# AI Strategy with Anthropic
MINER_STRATEGY=ai_reasoning
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Optional data sources
COINGECKO_API_KEY=your_coingecko_api_key_here
```


---

## 🔍 Troubleshooting

### Basic Monitoring
```bash
pm2 status              # Check all process status
pm2 logs                # View all logs in real-time
pm2 logs miner_1        # View specific process logs
pm2 monit               # Real-time monitoring dashboard
```

### Common Issues & Solutions

#### 1. Processes Won't Start
```bash
# Have you tried turning it off and on again
pm2 restart ecosystem.config.js

# Check if hotkeys exist and see helpful error messages
pm2 start ecosystem.config.js
```

#### 2. Check Environment
```bash
# Activate environment and test torch
source .venv/bin/activate
python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

# Should show: Torch version: 2.7.1+cpu, if it's CUDA version then reinstall it
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Verify Configuration
```bash
# Check your .env file
cat .env

# Should contain at minimum:
# WALLET_NAME=my_wallet
# HOTKEY_NAME=miner_1
# API_URL="https://api.subnet90.com"

# Compare with example
diff .env .env.example
```

#### 4. Registration Issues
```bash
# Check if hotkeys are registered on subnet
btcli subnets show --netuid 90
btcli wallet overview --wallet.name my_wallet
```

#### 5. Complete Reset
```bash
# Remove all processes
pm2 delete all

# Start fresh
pm2 start ecosystem.config.js
```

**See `pm2-commands-reference.md` for complete PM2 command reference.**

---

## 🎯 Success Indicators

When everything is working correctly, you should see:

```bash
pm2 status
# Should show processes as "online"
```

```bash
pm2 logs
# Look for:
# "Verification complete resolution=TRUE/FALSE confidence=XX.X"
# "Processing verification request statement=..."
```

---

## 📞 Support

**Join Discord**: [Bittensor - Subnet 90](https://discord.gg/QtuWjbk7aF) and contact `@_enzi_`

**Additional Documentation**:
- **`pm2-commands-reference.md`** - Complete PM2 command reference
- **`btcli-commands-reference.md`** - Bittensor CLI commands
- **`.env.example`** - Configuration template with all available options