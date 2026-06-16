

import os
import math
import difflib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)


# group effect on ranking ... 

TOURNAMENT_TIERS = {
    'FIFA World Cup': 5,
    'Confederations Cup': 4,
    'Copa América': 4,
    'UEFA Euro': 4,
    'AFC Asian Cup': 4,
    'African Cup of Nations': 4,
    'Gold Cup': 3,
    'UEFA Nations League': 3,
    'CONCACAF Nations League': 3,
    'UEFA Euro qualification': 3,
    'FIFA World Cup qualification': 3,
    'Copa América qualification': 3,
    'African Cup of Nations qualification': 3,
    'AFC Asian Cup qualification': 3,
    'CONCACAF Nations League qualification': 3,
    'Gold Cup qualification': 3,
    'EAFF Championship': 2,
    'WAFF Championship': 2,
    'Gulf Cup': 2,
    'CFU Caribbean Cup': 2,
    'Oceania Nations Cup': 2,
    'COSAFA Cup': 2,
    'CAFA Nations Cup': 2,
    'ABCS Tournament': 2,
    'Superclásico de las Américas': 2,
    'CONMEBOL–UEFA Cup of Champions': 2,
}

def get_tier(tournament: str) -> int:
    return TOURNAMENT_TIERS.get(tournament, 1)

def get_k_factor(tournament: str) -> int:
    tier = get_tier(tournament)
    return {5: 60, 4: 50, 3: 40, 2: 25, 1: 10}[tier]

CONFEDERATION_MAP = {
    **{t: 'UEFA' for t in [
        'Albania','Andorra','Armenia','Austria','Azerbaijan','Belarus','Belgium',
        'Bosnia and Herzegovina','Bulgaria','Croatia','Cyprus','Czech Republic',
        'Denmark','England','Estonia','Faroe Islands','Finland','France','Georgia',
        'Germany','Gibraltar','Greece','Hungary','Iceland','Ireland','Israel','Italy',
        'Kazakhstan','Kosovo','Latvia','Liechtenstein','Lithuania','Luxembourg',
        'Malta','Moldova','Montenegro','Netherlands','North Macedonia','Northern Ireland',
        'Norway','Poland','Portugal','Romania','Russia','San Marino','Scotland',
        'Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland','Turkey',
        'Ukraine','Wales',
    ]},
    **{t: 'CONMEBOL' for t in [
        'Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador','Paraguay',
        'Peru','Uruguay','Venezuela',
    ]},
    **{t: 'CONCACAF' for t in [
        'Antigua and Barbuda','Aruba','Bahamas','Barbados','Belize','Bermuda',
        'Canada','Cayman Islands','Costa Rica','Cuba','Curaçao','Dominican Republic',
        'El Salvador','Grenada','Guatemala','Haiti','Honduras','Jamaica','Mexico',
        'Montserrat','Nicaragua','Panama','Puerto Rico','Saint Kitts and Nevis',
        'Saint Lucia','Saint Vincent and the Grenadines','Trinidad and Tobago',
        'Turks and Caicos Islands','United States','United States Virgin Islands',
    ]},
    **{t: 'CAF' for t in [
        'Algeria','Angola','Benin','Botswana','Burkina Faso','Burundi','Cameroon',
        'Cape Verde','Central African Republic','Chad','Comoros','Congo',
        'Democratic Republic of the Congo','Djibouti','Egypt','Equatorial Guinea',
        'Eritrea','Eswatini','Ethiopia','Gabon','Gambia','Ghana','Guinea',
        'Guinea-Bissau','Ivory Coast','Kenya','Lesotho','Liberia','Libya',
        'Madagascar','Malawi','Mali','Mauritania','Mauritius','Morocco','Mozambique',
        'Namibia','Niger','Nigeria','Rwanda','São Tomé and Príncipe','Senegal',
        'Sierra Leone','Somalia','South Africa','South Sudan','Sudan','Tanzania',
        'Togo','Tunisia','Uganda','Zambia','Zimbabwe',
    ]},
    **{t: 'AFC' for t in [
        'Afghanistan','Australia','Bahrain','Bangladesh','Bhutan','Cambodia','China',
        'Chinese Taipei','Guam','Hong Kong','India','Indonesia','Iran','Iraq','Japan',
        'Jordan','Kuwait','Kyrgyzstan','Laos','Lebanon','Macau','Malaysia','Maldives',
        'Mongolia','Myanmar','Nepal','North Korea','Oman','Pakistan','Palestine',
        'Philippines','Qatar','Saudi Arabia','Singapore','South Korea','Sri Lanka',
        'Syria','Tajikistan','Thailand','Timor-Leste','Turkmenistan','United Arab Emirates',
        'Uzbekistan','Vietnam','Yemen',
    ]},
    **{t: 'OFC' for t in [
        'American Samoa','Cook Islands','Fiji','New Caledonia','New Zealand',
        'Papua New Guinea','Samoa','Solomon Islands','Tahiti','Tonga','Vanuatu',
    ]},
}
CONFEDERATIONS = ['AFC', 'CAF', 'CONCACAF', 'CONMEBOL', 'OFC', 'UEFA', 'OTHER']
CONF_TO_IDX = {c: i for i, c in enumerate(CONFEDERATIONS)}

def get_confederation(team: str) -> str:
    return CONFEDERATION_MAP.get(team, 'OTHER')


# data prep stuff
class WorldCupDataset(Dataset):
    FORM_WINDOW = 10

    def __init__(self, df: pd.DataFrame, scaler: dict = None, fit_scaler: bool = True):
        self.df = df.reset_index(drop=True)
        self.all_teams = sorted(set(df['Home_Team']) | set(df['Away_Team']))
        self.team_to_idx = {t: i for i, t in enumerate(self.all_teams)}

        elo = defaultdict(lambda: 1500.0)
        form = defaultdict(list)
        h2h  = defaultdict(list)

        rows = []
        for _, row in self.df.iterrows():
            h, a = row['Home_Team'], row['Away_Team']
            hg, ag = float(row['Home_Goals']), float(row['Away_Goals'])
            tourn = str(row['Tournament'])
            tier  = get_tier(tourn)
            k     = get_k_factor(tourn)

            h_elo, a_elo = elo[h], elo[a]

            h2h_key = tuple(sorted([h, a]))
            h2h_hist = h2h[h2h_key][-8:]

            h_form = form[h][-self.FORM_WINDOW:]
            a_form = form[a][-self.FORM_WINDOW:]

            def decay_mean(records, col):
                if not records:
                    return None
                weights = np.exp(np.linspace(-1, 0, len(records)))
                vals = [r[col] for r in records]
                return float(np.average(vals, weights=weights))

            def form_features(f):
                if not f:
                    return (1.3, 1.3, 1.3, 1.3, 4.0, 4.0, 50.0)
                return tuple(
                    decay_mean(f, i) if decay_mean(f, i) is not None else d
                    for i, d in enumerate([1.3, 1.3, 1.3, 1.3, 4.0, 4.0, 50.0])
                )

            h_ff = form_features(h_form)
            a_ff = form_features(a_form)

            if h2h_hist:
                first = h2h_key[0]
                if h == first:
                    h2h_gf  = np.mean([r[0] for r in h2h_hist])
                    h2h_ga  = np.mean([r[1] for r in h2h_hist])
                    h2h_xgf = np.mean([r[2] for r in h2h_hist])
                    h2h_xga = np.mean([r[3] for r in h2h_hist])
                else:
                    h2h_gf  = np.mean([r[1] for r in h2h_hist])
                    h2h_ga  = np.mean([r[0] for r in h2h_hist])
                    h2h_xgf = np.mean([r[3] for r in h2h_hist])
                    h2h_xga = np.mean([r[2] for r in h2h_hist])
                h2h_n = len(h2h_hist)
            else:
                h2h_gf = h2h_ga = h2h_xgf = h2h_xga = 1.3
                h2h_n = 0

            exp_h = 1.0 / (1 + 10 ** ((a_elo - h_elo) / 400.0))
            s_h   = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
            gd    = abs(hg - ag)
            gdm   = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 + (gd - 3) / 8.0))
            elo[h] += k * gdm * (s_h - exp_h)
            elo[a] += k * gdm * ((1 - s_h) - (1 - exp_h))

            form[h].append((hg, ag, float(row['xG_Home']), float(row['xG_Away']),
                            float(row['Shots_On_Target_Home']), float(row['Shots_On_Target_Away']),
                            float(row['Possession_Home'])))
            form[a].append((ag, hg, float(row['xG_Away']), float(row['xG_Home']),
                            float(row['Shots_On_Target_Away']), float(row['Shots_On_Target_Home']),
                            float(row['Possession_Away'])))
            h2h[h2h_key].append((
                hg if h == h2h_key[0] else ag,
                ag if h == h2h_key[0] else hg,
                float(row['xG_Home']) if h == h2h_key[0] else float(row['xG_Away']),
                float(row['xG_Away']) if h == h2h_key[0] else float(row['xG_Home']),
            ))

            rows.append({
                'home_idx': self.team_to_idx[h],
                'away_idx': self.team_to_idx[a],
                'home_conf': CONF_TO_IDX[get_confederation(h)],
                'away_conf': CONF_TO_IDX[get_confederation(a)],
                'tier': tier,
                'neutral': float(row['Neutral_Venue']),
                'h_elo': h_elo, 'a_elo': a_elo,
                'h_gf': h_ff[0], 'h_ga': h_ff[1],
                'h_xgf': h_ff[2], 'h_xga': h_ff[3],
                'h_sotf': h_ff[4], 'h_sota': h_ff[5], 'h_poss': h_ff[6],
                'a_gf': a_ff[0], 'a_ga': a_ff[1],
                'a_xgf': a_ff[2], 'a_xga': a_ff[3],
                'a_sotf': a_ff[4], 'a_sota': a_ff[5], 'a_poss': a_ff[6],
                'h2h_gf': h2h_gf, 'h2h_ga': h2h_ga,
                'h2h_xgf': h2h_xgf, 'h2h_xga': h2h_xga,
                'h2h_n': float(min(h2h_n, 8)),
                'home_goals': float(row['Home_Goals']),
                'away_goals': float(row['Away_Goals']),
                'weight': tier * 1.0,
            })

        self.records = rows
        self.elo_dict  = dict(elo)
        self.form_dict = dict(form)
        self.h2h_dict  = dict(h2h)

        CONT_KEYS = ['h_elo','a_elo','h_gf','h_ga','h_xgf','h_xga','h_sotf','h_sota','h_poss',
                     'a_gf','a_ga','a_xgf','a_xga','a_sotf','a_sota','a_poss',
                     'h2h_gf','h2h_ga','h2h_xgf','h2h_xga','h2h_n','tier']
        if fit_scaler:
            self.scaler = {}
            for k in CONT_KEYS:
                vals = [r[k] for r in self.records]
                self.scaler[f'{k}_mean'] = float(np.mean(vals))
                self.scaler[f'{k}_std']  = float(np.std(vals) + 1e-8)
        else:
            self.scaler = scaler
        self.CONT_KEYS = CONT_KEYS

    def scale(self, key, val):
        return (val - self.scaler[f'{key}_mean']) / self.scaler[f'{key}_std']

    def get_feature_vector(self, rec):
        cont = torch.tensor(
            [self.scale(k, rec[k]) for k in self.CONT_KEYS],
            dtype=torch.float32
        )
        return cont

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            'home_idx':  torch.tensor(r['home_idx'],  dtype=torch.long),
            'away_idx':  torch.tensor(r['away_idx'],  dtype=torch.long),
            'home_conf': torch.tensor(r['home_conf'], dtype=torch.long),
            'away_conf': torch.tensor(r['away_conf'], dtype=torch.long),
            'features':  self.get_feature_vector(r),
            'neutral':   torch.tensor(r['neutral'],   dtype=torch.float32),
            'home_goals':torch.tensor(min(r['home_goals'], 10.0), dtype=torch.float32),
            'away_goals':torch.tensor(min(r['away_goals'], 10.0), dtype=torch.float32),
            'weight':    torch.tensor(r['weight'],    dtype=torch.float32),
        }


# model bits
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class CrossAttention(nn.Module):
    """Let home and away feature vectors attend to each other."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.out   = nn.Linear(dim, dim)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, x, context):
        B, D = x.shape
        H = self.heads
        d = D // H
        Q = self.q(x).view(B, H, d)
        K = self.k(context).view(B, H, d)
        V = self.v(context).view(B, H, d)
        attn = (Q * K * self.scale).sum(-1, keepdim=True)
        attn = torch.sigmoid(attn)
        out  = (attn * V).view(B, D)
        return self.norm(x + self.out(out))


class WorldCupNet(nn.Module):
    EMBED_DIM = 64
    HIDDEN    = 256

    def __init__(self, num_teams: int, num_feats: int):
        super().__init__()
        n_conf = len(CONFEDERATIONS)

        self.team_atk = nn.Embedding(num_teams, self.EMBED_DIM)
        self.team_def = nn.Embedding(num_teams, self.EMBED_DIM)
        self.conf_emb = nn.Embedding(n_conf,   16)

        in_dim = self.EMBED_DIM * 4 + 16 * 2 + num_feats + 1
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.HIDDEN),
            nn.LayerNorm(self.HIDDEN),
            nn.GELU(),
        )

        stream_dim = self.HIDDEN // 2
        self.home_proj = nn.Linear(self.HIDDEN, stream_dim)
        self.away_proj = nn.Linear(self.HIDDEN, stream_dim)

        self.home_cross = CrossAttention(stream_dim, heads=4)
        self.away_cross = CrossAttention(stream_dim, heads=4)

        self.shared = nn.Sequential(
            ResidualBlock(self.HIDDEN, 0.25),
            ResidualBlock(self.HIDDEN, 0.20),
            ResidualBlock(self.HIDDEN, 0.15),
        )

        self.home_head = nn.Sequential(
            nn.Linear(self.HIDDEN, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        self.away_head = nn.Sequential(
            nn.Linear(self.HIDDEN, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.team_atk.weight, 0, 0.01)
        nn.init.normal_(self.team_def.weight, 0, 0.01)
        nn.init.normal_(self.conf_emb.weight, 0, 0.01)

    def forward(self, home_idx, away_idx, home_conf, away_conf, features, neutral):
        h_atk = self.team_atk(home_idx)
        h_def = self.team_def(home_idx)
        a_atk = self.team_atk(away_idx)
        a_def = self.team_def(away_idx)
        h_c   = self.conf_emb(home_conf)
        a_c   = self.conf_emb(away_conf)

        if neutral.dim() == 1:
            neutral = neutral.unsqueeze(-1)

        x = torch.cat([h_atk, h_def, a_atk, a_def, h_c, a_c, features, neutral], dim=-1)
        x = self.input_proj(x)

        h_stream = self.home_proj(x)
        a_stream = self.away_proj(x)

        h_stream = self.home_cross(h_stream, a_stream)
        a_stream = self.away_cross(a_stream, h_stream)

        fused = torch.cat([h_stream, a_stream], dim=-1)
        fused = self.shared(fused)

        home_lam = F.softplus(self.home_head(fused)).squeeze(-1) + 1e-4
        away_lam = F.softplus(self.away_head(fused)).squeeze(-1) + 1e-4
        return home_lam, away_lam


def weighted_nll_poisson(lam, goals, weight):
    """Negative log likelihood of Poisson(lam) weighted by sample importance."""
    nll = lam - goals * torch.log(lam + 1e-8) + torch.lgamma(goals + 1)
    return (nll * weight).sum() / weight.sum()


# Train loop

def train(dataset_train, dataset_val, max_epochs=2000, batch_size=128, lr=3e-4):
    loader_tr = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  drop_last=True)
    loader_va = DataLoader(dataset_val,   batch_size=256,        shuffle=False)

    num_feats = len(dataset_train.CONT_KEYS)
    model     = WorldCupNet(num_teams=len(dataset_train.all_teams), num_feats=num_feats)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)

    best_val_loss = float('inf')
    best_state    = None
    patience      = 200
    no_improve    = 0

    print(f"\n[Training] {len(dataset_train)} train / {len(dataset_val)} val samples")
    print(f"[Training] {len(dataset_train.all_teams)} teams | {num_feats} continuous features\n")

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss = 0.0
        for b in loader_tr:
            optimizer.zero_grad()
            hl, al = model(b['home_idx'], b['away_idx'],
                           b['home_conf'], b['away_conf'],
                           b['features'], b['neutral'])
            loss = weighted_nll_poisson(hl, b['home_goals'], b['weight']) + \
                   weighted_nll_poisson(al, b['away_goals'], b['weight'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b in loader_va:
                hl, al = model(b['home_idx'], b['away_idx'],
                               b['home_conf'], b['away_conf'],
                               b['features'], b['neutral'])
                va_loss += (weighted_nll_poisson(hl, b['home_goals'], b['weight']) +
                            weighted_nll_poisson(al, b['away_goals'], b['weight'])).item()

        avg_tr = tr_loss / len(loader_tr)
        avg_va = va_loss / len(loader_va)

        if avg_va < best_val_loss - 1e-4:
            best_val_loss = avg_va
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:04d} | Train: {avg_tr:.4f} | Val: {avg_va:.4f} | Best Val: {best_val_loss:.4f}")

        if no_improve >= patience:
            print(f"\n[Early Stop] Triggered at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break

    model.load_state_dict(best_state)
    return model


def _get_team_record(dataset, team):
    """Build a synthetic record for inference using latest Elo + form + H2H."""
    elo = dataset.elo_dict.get(team, 1500.0)
    f   = dataset.form_dict.get(team, [])[-dataset.FORM_WINDOW:]

    def dw(col, default):
        if not f: return default
        w = np.exp(np.linspace(-1, 0, len(f)))
        return float(np.average([r[col] for r in f], weights=w))

    return {
        'h_elo': elo, 'a_elo': elo,
        'h_gf': dw(0, 1.3), 'h_ga': dw(1, 1.3),
        'h_xgf': dw(2, 1.3), 'h_xga': dw(3, 1.3),
        'h_sotf': dw(4, 4.0), 'h_sota': dw(5, 4.0), 'h_poss': dw(6, 50.0),
        'a_gf': dw(0, 1.3), 'a_ga': dw(1, 1.3),
        'a_xgf': dw(2, 1.3), 'a_xga': dw(3, 1.3),
        'a_sotf': dw(4, 4.0), 'a_sota': dw(5, 4.0), 'a_poss': dw(6, 50.0),
        'h2h_gf': 1.3, 'h2h_ga': 1.3, 'h2h_xgf': 1.3, 'h2h_xga': 1.3, 'h2h_n': 0.0,
        'tier': 5,
    }


def _lambda_for_matchup(model, dataset, team_h, team_a, neutral_val):
    """Return (lambda_home, lambda_away) from the network."""
    rec_h = _get_team_record(dataset, team_h)
    rec_a = _get_team_record(dataset, team_a)

    merged = {
        'h_elo': rec_h['h_elo'], 'a_elo': rec_a['a_elo'],
        'h_gf': rec_h['h_gf'], 'h_ga': rec_h['h_ga'],
        'h_xgf': rec_h['h_xgf'], 'h_xga': rec_h['h_xga'],
        'h_sotf': rec_h['h_sotf'], 'h_sota': rec_h['h_sota'], 'h_poss': rec_h['h_poss'],
        'a_gf': rec_a['a_gf'], 'a_ga': rec_a['a_ga'],
        'a_xgf': rec_a['a_xgf'], 'a_xga': rec_a['a_xga'],
        'a_sotf': rec_a['a_sotf'], 'a_sota': rec_a['a_sota'], 'a_poss': rec_a['a_poss'],
        'tier': 5,
    }

    h2h_key = tuple(sorted([team_h, team_a]))
    h2h_hist = dataset.h2h_dict.get(h2h_key, [])[-8:]
    if h2h_hist:
        first = h2h_key[0]
        if team_h == first:
            merged['h2h_gf']  = np.mean([r[0] for r in h2h_hist])
            merged['h2h_ga']  = np.mean([r[1] for r in h2h_hist])
            merged['h2h_xgf'] = np.mean([r[2] for r in h2h_hist])
            merged['h2h_xga'] = np.mean([r[3] for r in h2h_hist])
        else:
            merged['h2h_gf']  = np.mean([r[1] for r in h2h_hist])
            merged['h2h_ga']  = np.mean([r[0] for r in h2h_hist])
            merged['h2h_xgf'] = np.mean([r[3] for r in h2h_hist])
            merged['h2h_xga'] = np.mean([r[2] for r in h2h_hist])
        merged['h2h_n'] = float(len(h2h_hist))
    else:
        merged['h2h_gf'] = merged['h2h_ga'] = merged['h2h_xgf'] = merged['h2h_xga'] = 1.3
        merged['h2h_n'] = 0.0

    feat = torch.tensor(
        [dataset.scale(k, merged[k]) for k in dataset.CONT_KEYS],
        dtype=torch.float32
    ).unsqueeze(0)

    h_idx  = torch.tensor([dataset.team_to_idx[team_h]], dtype=torch.long)
    a_idx  = torch.tensor([dataset.team_to_idx[team_a]], dtype=torch.long)
    h_conf = torch.tensor([CONF_TO_IDX[get_confederation(team_h)]], dtype=torch.long)
    a_conf = torch.tensor([CONF_TO_IDX[get_confederation(team_a)]], dtype=torch.long)
    neut   = torch.tensor([[neutral_val]], dtype=torch.float32)

    with torch.no_grad():
        hl, al = model(h_idx, a_idx, h_conf, a_conf, feat, neut)
    return hl.item(), al.item()


#match sim

def predict_match(model, dataset, team1, team2, n_sim=50_000, max_goals=10):
    model.eval()

    HOST_TEAMS = {'united states', 'usa', 'us', 'mexico', 'canada'}
    t1_is_host = team1.lower() in HOST_TEAMS
    t2_is_host = team2.lower() in HOST_TEAMS

    if not t1_is_host and not t2_is_host:
        lam1_a, lam2_a = _lambda_for_matchup(model, dataset, team1, team2, neutral_val=1.0)
        lam2_b, lam1_b = _lambda_for_matchup(model, dataset, team2, team1, neutral_val=1.0)
        lam1 = (lam1_a + lam1_b) / 2
        lam2 = (lam2_a + lam2_b) / 2
        venue_note = "Neutral venue"
    elif t1_is_host and not t2_is_host:
        lam1, lam2 = _lambda_for_matchup(model, dataset, team1, team2, neutral_val=0.0)
        venue_note = f"Home advantage  {team1}"
    elif t2_is_host and not t1_is_host:
        lam2, lam1 = _lambda_for_matchup(model, dataset, team2, team1, neutral_val=0.0)
        venue_note = f"Home advantage  {team2}"
    else:
        lam1_a, lam2_a = _lambda_for_matchup(model, dataset, team1, team2, neutral_val=1.0)
        lam2_b, lam1_b = _lambda_for_matchup(model, dataset, team2, team1, neutral_val=1.0)
        lam1 = (lam1_a + lam1_b) / 2
        lam2 = (lam2_a + lam2_b) / 2
        venue_note = "Both are hosts ... treated as neutral"

    g1 = np.random.poisson(lam1, n_sim)
    g2 = np.random.poisson(lam2, n_sim)

    p_win1  = float(np.mean(g1 > g2))
    p_draw  = float(np.mean(g1 == g2))
    p_win2  = float(np.mean(g1 < g2))

    p1 = [math.exp(-lam1) * lam1**i / math.factorial(i) for i in range(max_goals + 1)]
    p2 = [math.exp(-lam2) * lam2**i / math.factorial(i) for i in range(max_goals + 1)]
    mat = np.outer(p1, p2)
    rho = -0.10
    mat[0,0] *= max(0, 1 - lam1 * lam2 * rho)
    mat[1,0] *= max(0, 1 + lam1 * rho)
    mat[0,1] *= max(0, 1 + lam2 * rho)
    mat[1,1] *= max(0, 1 - rho)
    mat /= mat.sum()
    modal_r, modal_c = np.unravel_index(np.argmax(mat), mat.shape)

    elo1 = dataset.elo_dict.get(team1, 1500)
    elo2 = dataset.elo_dict.get(team2, 1500)

    h2h_key  = tuple(sorted([team1, team2]))
    h2h_hist = dataset.h2h_dict.get(h2h_key, [])

    print("\n" + "═"*62)
    print(f"  {team1.upper()}  vs  {team2.upper()}")
    print("═"*62)
    print(f"  Venue          : {venue_note}")
    print(f"  Elo Ratings    : {team1} {elo1:.0f}  |  {team2} {elo2:.0f}")
    print(f"  form      : {_get_team_record(dataset,team1)['h_xgf']:.2f} vs "
          f"{_get_team_record(dataset,team2)['h_xgf']:.2f}")
    print(f"  Expected Goals : {lam1:.2f} – {lam2:.2f}")
    print(f"  Most Likely    : {modal_r} – {modal_c}  "
          f"(p={mat[modal_r, modal_c]*100:.1f}%)")
    print(f"  H2H games used : {len(h2h_hist)}")
    print("─"*62)
    print(f"  {team1:<22} Win : {p_win1*100:6.2f}%")
    print(f"  {'Draw':<22}     : {p_draw*100:6.2f}%")
    print(f"  {team2:<22} Win : {p_win2*100:6.2f}%")
    print("═"*62)

    flat = [(mat[i,j], i, j) for i in range(max_goals+1) for j in range(max_goals+1)]
    flat.sort(reverse=True)
    print("  Top score lines:")
    for prob, i, j in flat[:5]:
        print(f"    {i}–{j}  →  {prob*100:.2f}%")
    print("═"*62)


def get_valid_team(prompt, teams):
    while True:
        inp = input(prompt).strip()
        if inp.lower() == 'exit':
            return 'exit'
        for t in teams:
            if t.lower() == inp.lower():
                return t
        matches = difflib.get_close_matches(inp, teams, n=3, cutoff=0.45)
        if matches:
            if len(matches) == 1:
                print(f"  Matched: {matches[0]}")
                return matches[0]
            print(f"  Did you mean: {', '.join(matches)}?")
            continue
        print(f"  [!] '{inp}' not found. Try again or type 'exit'.")


# Main entry
if __name__ == '__main__':
    EXCEL      = 'Advanced_WorldCup_Stats_2014_2026.xlsx'
    MODEL_FILE = 'worldcup_model.pt'
    SCALER_FILE= 'worldcup_scaler.npy'

    print("[Loading] Reading dataset …")
    raw = pd.read_excel(EXCEL)
    raw['Home_Team'] = raw['Home_Team'].str.strip()
    raw['Away_Team'] = raw['Away_Team'].str.strip()
    raw['Date']      = pd.to_datetime(raw['Date'])
    raw = raw.sort_values('Date').reset_index(drop=True)

    split = int(len(raw) * 0.90)
    df_tr = raw.iloc[:split]
    df_va = raw.iloc[split:]
    print(f"[Split] Train : {len(df_tr)} | Val: {len(df_va)}")

    dataset_train = WorldCupDataset(df_tr, fit_scaler=True)
    dataset_val   = WorldCupDataset(df_va, scaler=dataset_train.scaler, fit_scaler=False)
    dataset_val.all_teams  = dataset_train.all_teams
    dataset_val.team_to_idx= dataset_train.team_to_idx
    dataset_val.elo_dict   = dataset_train.elo_dict
    dataset_val.form_dict  = dataset_train.form_dict
    dataset_val.h2h_dict   = dataset_train.h2h_dict

    print("[Building] Full dataset for inference state …")
    dataset_full = WorldCupDataset(raw, fit_scaler=True)

    num_feats = len(dataset_train.CONT_KEYS)

    if os.path.exists(MODEL_FILE):
        print(f"[Loading] Found saved model: {MODEL_FILE}")
        model = WorldCupNet(num_teams=len(dataset_full.all_teams), num_feats=num_feats)
        model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
    else:
        print("[Training] No saved model found ... training from scratch …")
        model = train(dataset_train, dataset_val, max_epochs=2000, batch_size=128, lr=3e-4)
        torch.save(model.state_dict(), MODEL_FILE)
        print(f"[Saved] Model  {MODEL_FILE}")

    print("\n═══ World Cup Predictor — READY ═══")
    print("Type team names to predict. Enter 'exit' to quit.\n")

    while True:
        t1 = get_valid_team("Team 1 : ", dataset_full.all_teams)
        if t1 == 'exit': break
        t2 = get_valid_team("Team 2: ", dataset_full.all_teams)
        if t2 == 'exit': break
        if t1 == t2:
            print("  [!] Teams must be different.")
            continue
        predict_match(model, dataset_full, t1, t2)
