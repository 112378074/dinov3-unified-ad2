# ARCHITECTURE v2 — 降低誤檢/漏檢 + 邏輯瑕疵路徑 (2026-07-05, 對抗審查更正版)

來源: 使用者指認的失敗案例 (walnuts/004_shift_3, vial/008_regular 邏輯瑕疵; wallplugs/007_shift_2, 004_shift_3 光照變體) → 三家族分析 (CSV+面板) → 架構提案 → 對抗審查 (10 組件: 2 KEEP, 8 FIX) → 本更正版。

## 失敗成因總表

# ARCHITECTURE v2（audit 修正版）

基線 = deployed v3（AUPRO 72.46 / AUROC 85.05 / SegF1 63.89 / ClassF1 83.65，4 目標全達）。所有新組件 ADDITIVE、per-cat opt-in、gate 不過即回滾；float base map 與 binaries-from-base-maps 規則不動；採用/棄用決策只依 VarVal，test_public 只允許軌級一次性驗證。

## 1. 失效成因總表（CSV + 目檢雙重驗證）

| 家族 / cat | 代表案例 | 機制（一行） |
|---|---|---|
| LOGICAL：真 float 沉默 | vial 008_regular（下半瓶身內容/液位）、fj 006/011（內嵌物）、walnuts 004_shift_3（開殼碎片） | 每個局部 patch 都落在正常流形內，異常只存在於位置條件化的「組成」——兩個 PATCH-LOCAL 分支結構性看不見，map 擴散無自信區 |
| LOGICAL（改判→binary-chain） | walnuts 009（6/6 變體 peak_in_gt=1、detected=0）、014（3/6） | float 其實已在 GT 上開火，miss 發生在二值化鏈（門檻/幅度），非 patch-local 盲區——歸帳給二值化路，不歸 Branch C |
| PHOTOMETRIC：動態範圍塌縮 | wallplugs 007_shift_2 | 暗變體下 float map 全域抑制（per-image std→0），2px GT 不可見——表徵缺口，非門檻問題 |
| PHOTOMETRIC：陰影 FP flood | wallplugs 004_shift_3、walnuts goods 8/8 全在 shift_1/3 | cast-shadow 邊界 = out-of-bank 的正常外觀 → 大量熱點全不在 GT 上 |
| PHOTOMETRIC：specular（不可合成） | can goods 44/78 FP、共 59,801px | 姿態/光照 shift 下 foil/雷射標籤 sheen 整體變貌；已驗證 2D 增強無法合成——期望一律記「未知」 |
| can（bads，寫死） | 83/84 miss、僅 1 個 peak-in-GT | 4.5px 刮痕 sub-token（/16 解析度死路，已驗證）——明確不承諾 |
| wallplugs | 67 miss = 51 零 float 響應 + 16 peak-in-GT | 51 案 camouflage 寫死；可回收池只有 16 案 + 變體塌縮恢復 |
| sheet_metal | 31 miss 中 28 個 peak-in-GT；goods 0/24 FP | 絕對門檻過保守 + 003_regular 垂直 stitch 縫 artifact 拉高 ambient floor |
| fabric | 61 miss 中 21 個 peak-in-GT；goods 2/72（48px） | 小峰低於全域門檻，FP 預算幾乎全空 |
| rice | goods 0/42 FP；同 specimen 跨變體偵測翻轉 | per-image ambient floor 隨變體漂移 × 絕對門檻 |
| fruit_jelly | 5/15 specimen 全漏；goods 1/20（9px） | 杯內局部對比不足 + rim 干擾；FP 預算全空 |
| vial | 105/105 全偵測，但 bads mean fp 628.7px、fp>gt 39/105 | SegF1 損失來自 spill/標籤 FP 而非漏檢；008/013 定位擴散 |
| walnuts | 30 miss 中 17 個 peak-in-GT；goods FP 僅 824px | 混合：binary-chain（大宗）+ 真 logical 碎片（少數）+ shift_1/3 陰影弧 |
| 系統性 | 全 cat | validation 只有 regular（陷阱已咬 4 提案）；部署常數為 test-tuned（legitimacy audit）——任何校準必須走增強後的 VarVal，且重校成本入帳 |

## 更正後組件

## 2. 修正後組件清單（含指標路由 / 資料計畫 / gate）

**共同規則**：二值化改動 float-path hash 自動檢查不變；每個 miss/FP 池只入帳一次（stage-logging 歸因後決定唯一 owner）；所有超參只在 VarVal 上定。

### K1. VarVal 變體增強校準集（修正版——不再是單一 BLOCKER）
- 機制：held-out train normals + validation normals 套 photometric（gamma/gain/exposure，忠實可合成）＋合成定向軟陰影＋crop-offset/shift＋既有合成異常＋SynLogical 擾動（挖空/置換/液面平移/大區塊交換）。
- 修正：(i) 建構規格**預先登記、最多一輪修訂**——嚴禁把增強參數迭代到「重現 test 統計」；(ii) 保真目標只限 **normal-side 聚合方向性**（good-image FP 趨勢），rice 條目改用 held-normal floor 統計，不用 test-GT 相依的偵測翻轉；(iii) **can-specular 預先登記為不可重現、排除在判準之外**（已驗證不可合成）；(iv) 降級規則自洽化：E10a 若失敗，**photometric-faithful 類 gate 保留效力，所有 shift/shadow 相依 gate（不分組件）降為 unsupported**——不再是「凍 7/8/9 留 2/3/5/6」的假隔離；(v) 繼承逐常數守門＋SAFE 回退；「VarVal 常數取代 test-tuned 常數 → 分數預期下修」列為明文預算負項。
- 路由：不直接餵指標；是其餘 gate 的先決條件。資料：零 test 影像。
- Gate（E10a 修正版）：per-cat/per-現象判準，僅 normal-side 方向性（如 wallplugs 暗變體抑制、walnuts 陰影弧 FP）。

### K2. Branch C：低容量特徵 AE（logical 通路，縮圍版）
- 機制：frozen DINOv3 特徵圖上窄 bottleneck AE（32/64/128 掃描），位置+全域條件化重建正常組成，殘差圖給 float。
- 修正：(i) **walnuts 標的縮到 float 真正沉默的組成異常**（004_shift_3 碎片、空殼未著火區）；009/014 是 binary-chain 誤失，**不得入帳 C**，須待 K0 stage logging 歸因；(ii) vial 105/105 已偵測 → C 對 vial **只路由 SegF1/AUPRO 定位**，權重保守化＋vial-AUPRO 無回歸 gate（vial float 是強項）；(iii) 一次性 test 驗證附**預先登記反事實**：採用與否只依 VarVal，命名影像未翻轉不得推翻。
- 路由：float 第三通道（vial/fj/walnuts）→ AUPRO；經 fused binary → SegF1；通道 max → ClassF1。
- 資料：train normals，訓練資料**強制 photometric+shift+shadow 增強**（否則變體殘差爆炸=第一死因）；容量/權重只在 VarVal 選。
- Gate（E11）：held-normal VarVal 殘差 P99 < SynLogical 殘差中位數 ÷2（≥2× 對比）；fuse 後 VarVal-synthetic AUPRO ≥+2 且變體 normals FP 不升；vial-AUPRO 不退。

### K3. Branch C2：物件級 pooled-embedding kNN（改為加法項）
- 機制：walnuts 每顆核桃 masked mean-pool → train-normal 物件 bank kNN；vial 位置條件化水平帶 pooling 對 normal 帶 bank kNN（bank-referenced，非 self-normalized——確實避開 per-image normalization 負結果）。
- 修正：(i) 分數以 **validation-加權加法/max 項廣播**，非「塗回整顆 segment」硬替換（合併實例假高分會注入大片 float FP）；(ii) 接觸/重疊核 **watershed split＋人工 QA**；walnuts-AUPRO 無回歸 gate（57.5>ISVL++ 53 是強項）；(iii) vial 帶以 **product-ROI 相對座標**定義（test 有 shift_1–4，絕對座標會整排 FP）；vial 收益只路由 SegF1/AUPRO；(iv) **預先登記註銷條件**：train normals 液位自然變異若把 008 型異常收進 normal 帶流形 → 此路廢棄；(v) 009/014 與 K2/K7 三重入帳問題——stage logging 歸因後**單一入帳**。
- 資料：前景 connected-components + SAM3 僅偵測用途（已驗證合法）；物件 bank 含 shift/photometric 增強；只上 walnuts/vial/fj，**絕不 wallplugs**。
- Gate（E12）：VarVal held-normal 物件 FP <2%；SynLogical 物件命中 ≥80%；黏連抽查 <10%。

### K4. 全域 embedding kNN 影像分數輔助（ClassF1-only）
- 修正：**刪除 AUROC 入帳**——AU-ROC 是 continuous-map 指標（本專案已驗證），image-score 融合動不了它。只記 ClassF1。
- 資料：train-normal CLS/pooled bank，必含變體增強。Gate：VarVal 增強 normals image-FP 不升，否則直接棄用。

### K5. V1-Lambertian：陰影/context 增強 membank 重建（can 降級）
- 可信半（保留）：wallplugs/walnuts 陰影/context 增強（近朗伯場景），aug_every 管線＋同款陰影負樣本鏡射進 Branch B synthesis；coreset 稀釋監控（fabric/rice/walnuts 不退——weak-bank artifact 前科）；**membank 崩潰域警語**：bank 增長需避開 crop640+bank500+ROI 已知 native crash 域。
- 不可信半（降級）：can pose-jitter 真實 crop 只有 regular 光照下的 specular 外觀——**期望記「未知」**，不得覆寫；原「can 12/12 shift-FP 單調性被打破」gate 是 test-informed，**改 VarVal-only**（而 VarVal 不能表徵 can-shift → can 部分實際不可評）。V1-can 降為綁 K12（can/3）的廉價附帶實驗。
- 路由：float FP 下降 → AUPRO+AUROC；經 binary → SegF1；good 誤報下降 → ClassF1。
- Gate（E13a，VarVal-only）：good FP 中位數 −≥50% 且 synthetic 偵測率不降；fabric/rice/walnuts AUPRO 不退；過 gate 後 pen/lam 於 VarVal 重校（成本入帳）。

### K6. D1：wallplugs 變體增強蒸餾頭（KEEP，帳目微修）
- 機制：已證配方（out-of-bank + synthetic-anomaly-aug + photometric-aug；E2d walnuts 111%、E3c fabric/rice 95%）＋shift/shadow 增強，攻暗變體動態範圍塌縮。
- 修正：teacher 在 **51/67**（非 49/67）零響應 camouflage 上蒸餾不可能創造訊號——期望只記**變體塌縮恢復＋16 個 peak-in-GT 可回收案**；配方在 wallplugs 未證過（fj 70% pending）。
- Gate（E13b）：regular VarVal ≥95% teacher；暗變體 per-image std 恢復 ≥ regular 的 50%；**完整管線（hires 2.0＋mask_fusion）下測**；任一指標退步單 cat 回滾。

### K7. Binarization v2：局部對比二值化（binary path only）
- 機制：sheet_metal top-hat/local-floor、fabric 小峰提升、rice per-image floor 減法（rice-only/binary-only/其他 cat bit-identical 驗證）、fj 杯內 peak-to-local-ambient＋rim mask、walnuts 小峰提升。FP 預算全部 CSV 驗證屬實（rice 0/42、sheet_metal 0/24、fabric 48px、fj 9px）。
- 修正：(i) **P2 substrate 決策先行**（synthetic-val 導出 float-vs-distance），否則 miss 池重校＋重複入帳；(ii) walnuts/wallplugs 小峰**種子改讀 hires fused-float 峰值**（base-res substrate 同位置可能無幅度）；(iii) sheet_metal 延長/離心率豁免須繼承 **seam-band 排除＋方向 gate**；(iv) 相對 27 案為「取代」非「疊加」，每 miss 池單一入帳。
- 路由：SegF1 直接；偵測翻轉 → ClassF1；AUPRO/AUROC 依構造零影響（float hash 檢查）。
- Gate（E14a–d）：VarVal synthetic SegF1 上升；VarVal 增強 good FP ≤ 各 cat 明示 px 預算；float hash 不變；sheet_metal 先過 K10 縫診斷。

### K8. V2：變體叢集條件門檻偏移（binary-only；float 試驗刪除）
- 保留：can/wallplugs binary 版——叢集級粗粒度、held-normal-referenced、影像內容特徵（無檔名）、regular bit-identical、vial 永不套用。
- 修正：**float 試驗刪除**——can 的 VarVal 量不到 specular shift（gate 形同虛設），且叢集偏移破壞跨影像分數可比性＝vial oracle 21.9 教訓的一般化（per-image 與 per-cluster 只差粒度）。
- Gate（E14e）：VarVal 叢集指派 ≥90%；SegF1/ClassF1 於 VarVal 上升；regular bit-identical。

### K9. 陰影邊界 FP gate（KEEP，範圍/帳目微修）
- 修正：wallplugs 範圍改 **shift_1/2/3（shift_2 為主）**（CSV：4 張 good FP 在 shift_1）；與 wallplugs 觸邊 blob gate / walnuts always-on 防衛**每 cat 擇一機制、FP 池單一入帳**；goods FP 僅 824/1,236px → 期望帳目主路由 **float ranking（AUPRO/AUROC）＋ClassF1**，SegF1 影響微小。
- Gate（E15）：VarVal 陰影 FP −≥70%；defect-on-shadow-boundary 保留 ≥90%；完整管線（gain 1.4＋mask_fusion）下測；不碰 can。

### K10. sheet_metal PoU/stitch 縫線修復（KEEP）
- 註記：003_regular 診斷可、調參禁；修復參數由 tile 幾何/VarVal 導出；只修縫不動解析度（hires 69→51 前科）。
- Gate：縫位置 float 能量與 tile 邊界對齊才確認 artifact；修後 goods 維持 0 FP、regular AUPRO 不退。

### K11.（新增，補幽靈帳）vial FP 抑制：標籤抑制＋保守氣泡場
- 機制：vial/1 標籤區域 FP 抑制（天花板 +0.5~1 cat）＋vial/2 氣泡場模式保守版——vial SegF1 最大 headroom（bads mean fp 628.7px、fp>gt 39/105）的**唯一 owner**。
- 路由：SegF1 主；binary path。Gate：VarVal photometric normals FP 下降、vial-AUPRO 不退、float hash 不變。

### K12.（新增）can/3 幾何包絡抑制器（can 條目重排首位）
- 機制：面積/aspect/包絡幾何特徵 FP 抑制——對外觀變化穩健，27 案評為 can 最可靠槓桿。can 採用順序：**can/3 → K8 binary → V1-can（廉價、期望未知）**。
- 路由：binary → SegF1/ClassF1；float 版僅在幾何規則（外觀無關）下小試。Gate：VarVal-可量部分過 gate；specular-shift 部分明文標注殘差缺口。

## 分階段實驗計畫

## 3. 階段化實驗計畫（便宜→昂貴，全部 gated；P0–P1 可與 E3c 蒸餾續跑並行）

**test 紀律（全計畫）**：採用/棄用只依 VarVal gate；test_public 峰視收斂為**軌級各一次**（T1 logical / T2 variant / T3 binary）＋E16 一次全量；回滾只允許**預先登記的災難性門檻（單 cat 單 metric −2.0 以上）**，一旦觸發回滾，最終配置文件必須標注 test_public-informed。

### P0 — Tier-1 廉價診斷（立即，CPU/輕 GPU，與 E3c 並行；零風險，改寫後續範圍）
- P0a walnuts **stage logging**：float→binary 鏈逐階歸因 30 個 miss（特別是 009/014）——直接決定 K2/K3/K7 的 walnuts 入帳分配。
- P0b rice Step1 階段歸因；P0c fj/wallplugs **hires A-map 重用稽核**（可能讓 K7 的 hires-seed 成本歸零）；P0d vial 載具列稽核。

### P1 — VarVal 建置（與 P0 並行；photometric 核先行）
- P1a photometric 增強核（忠實類）→ 立即解鎖 P2。
- P1b shift/shadow 增強＋SynLogical 擾動集（目檢合理性）。規格預先登記、最多一輪修訂。
- P1c E10a 修正版保真檢查：per-cat normal-side 方向性判準；can-specular 預先登記排除。失敗 → 只降級 shift/shadow 相依 gate 為 unsupported，photometric 類照常。

### P2 — 軌 T3 BINARY（便宜、證據最強；只需 P1a＋合成缺陷即可開跑）
- E14-0 substrate 決策（synthetic-val float-vs-distance）→ E14a sheet_metal top-hat（先過 K10 縫診斷）→ E14b fabric → E14c rice floor 減法（其他 cat bit-identical 驗證）→ E14d fj → E14e K8 binary 叢集門檻（can/wallplugs）→ K11 vial 標籤抑制＋氣泡場 → K12 can/3 幾何包絡。
- E15 陰影邊界 gate（需 P1b 陰影增強；完整管線下測）。共同 gate：float hash 不變。

### P3 — 軌 T2 VARIANT（GPU 重；E3c 跑完後接檔）
- E13b K6 wallplugs D1 蒸餾（已證配方＋shift/shadow 增強；完整管線下測）。
- E13a K5 V1-Lambertian bank 重建（wallplugs/walnuts；coreset 稀釋監控＋crash 域警語）；過 gate 後 pen/lam VarVal 重校（成本入帳）。
- V1-can 廉價附帶實驗（期望未知，綁 K12 之後）。

### P4 — 軌 T1 LOGICAL（最重；範圍依 P0a 歸因結果收縮）
- E11 K2 AE 容量掃描（32/64/128；walnuts/vial/fj）→ 融合權重 VarVal 擬合。
- E12 K3 物件 kNN（walnuts 加法項＋watershed QA；vial ROI-相對帶；註銷條件預先登記）。
- E12c K4 全域 kNN（cheap，ClassF1-only，gate 不過即棄）。
- 軌級一次性 test_public 驗證（T1 一次）。

### P5 — E16 整合
- 只組裝過 gate 組件成 per-cat adoption matrix（例：walnuts=K2縮圍+K3+K5+K9；vial=K2+K3帶版+K11；wallplugs=K6+K8+K9；can=K12→K8→V1-can；sheet_metal/fabric/rice/fj=K7 各條目＋K10）。未動 cat 凍結 v3 常數。單次 test_public 全量跑，報告 honest per-metric delta。
- 明確不做：can 4.5px 刮痕、seg-head synth for can、per-image score normalization、SAM3-fg-for-synthesis、bank-overlap 蒸餾、K8 float 版、vial 任何 adaptive 門檻。

## 審查更正記錄

## 4. Audit 變更紀錄（fixed / killed / kept 與理由）

判定統計：KEEP 3、FIX 9、KILL 0（整組件）；**條目級刪除 3 處**；**新增組件 2＋診斷前置 4**。

| 項目 | 判定 | 變更內容與理由 |
|---|---|---|
| K1 VarVal / E10a | FIX | BLOCKER 框架→per-cat 判準；保真目標剔除 test-GT 相依現象（rice 004/014 偵測翻轉→held-normal floor 統計），防止把增強生成器擬合到 test；can-specular 預先登記不可重現並排除（已驗證不可合成，原判準注定失敗會無辜凍結 threshold 軌）；降級分割改為「photometric gate 保留 / shift-shadow gate 全降 unsupported」（原「凍 7/8/9 留 2/3/5/6」不自洽——後者 gate 同在 VarVal 上量）；重校常數下修列預算負項 |
| K2 Branch C AE | FIX | walnuts 009/014 除帳（CSV：009 六變體 peak_in_gt=1＝binary-chain 誤失，非 patch-local 盲區）；標的縮至真 float-沉默案例；vial 除偵測帳（105/105 已偵測）只留定位；vial 權重保守化＋AUPRO 無回歸 gate；test 驗證附預先登記反事實 |
| K3 C2 物件 kNN | FIX | 「塗回整顆 segment」→validation-加權加法/max 廣播（防合併實例假高分注入 float FP）；補 walnuts-AUPRO 無回歸 gate、watershed＋QA；vial 帶改 product-ROI 相對座標（test 有 shift_1–4）；vial 收益改路由定位；預先登記註銷條件；009/014 三重入帳→歸因後單一入帳 |
| K4 全域 kNN | FIX | **刪除 AUROC 入帳**（AU-ROC=continuous-map 指標，image-score 融合動不了）；降為 ClassF1-only |
| K5 V1 | FIX | 拆半：Lambertian（wallplugs/walnuts）可信保留＋crash 域警語；can 半降級——specular 不可合成、原 gate 是 test-informed（test goods 上量單調性）改 VarVal-only、期望記未知；can 條目重排 can/3 先行（原案完全遺漏最可靠槓桿） |
| K6 D1 | KEEP | 微修：51/67（非 49/67）；期望只記變體塌縮恢復＋16 peak-in-GT，不暗示回收 51 camouflage 案；配方 wallplugs 未證註記 |
| K7 Binarization v2 | FIX | P2 substrate 決策先行；walnuts/wallplugs 種子改讀 hires fused-float 峰；sheet_metal 豁免繼承 seam-band 排除＋方向 gate；「取代非疊加」單一入帳 |
| K8 V2 叢集門檻 | FIX | binary 版保留；**float 試驗刪除**（can VarVal 量不到 specular shift＝gate 虛設；叢集偏移破壞跨影像可比性＝vial oracle 21.9 教訓一般化） |
| K9 陰影 gate | KEEP | 微修：wallplugs 範圍 shift_1/2/3（CSV 驗證 4 張在 shift_1）；每 cat 擇一機制單一入帳；帳目改路由 float ranking/ClassF1（goods FP px 太小撐不起 SegF1 主張） |
| K10 seam 修復 | KEEP | 註記：003_regular 診斷可調參禁；修復參數由 tile 幾何/VarVal 導出 |
| K11 vial 抑制 | 新增 | 補 SegF1 最大宣稱增益的**幽靈帳**（628.7px/39:105 池原本無任何組件負責——K7 無 vial 條目、K8 vial 永不套用、K2/K3 是加訊號非抑制） |
| K12 can/3 包絡 | 新增 | 27 案評 can 最可靠槓桿，原 adoption matrix 遺漏；can 條目重排 can/3→K8→V1-can |
| experiment_plan | FIX | ~10 個組件級一次性 test 峰視＋「test 退步即回滾」＝adopt-on-test 同族（SAM-adopt/can-scale 前科）→收斂為軌級各一次＋E16 一次＋預先登記災難性回滾門檻（觸發即標注 test_public-informed）；補排 Tier-1 廉價診斷於三軌之前；順序倒置修正——Binarization v2 提前（只需 photometric VarVal），重機件後行 |
| expected_gains | FIX | 刪 vial 幽靈項（改由 K11 承接）；刪全域 kNN 的 AUROC 項；V1-can「信心中」降未知；補 pen/lam/gain VarVal 重校＝毛額轉淨額的明文扣除；事實修正 51/67；範圍全面下修（見第 5 節） |

## 誠實預期增益

## 5. 誠實預期增益表（audit 修正後；全為「gate 過＋淨額」口徑）

| 指標 | v3 基線 | 誠實範圍 | 主要來源（單一入帳後） | Caveat |
|---|---|---|---|---|
| AUPRO | 72.46 | **+1.0 ~ +2.5**（→ ~73.5–75） | walnuts K3 加法項＋K9 goods ranking；vial K2 定位（008/013）；fj K2/K3；sheet_metal K10 縫修；can K12 小幅 | can 貢獻=未知（specular 不可合成，V1-can 期望未知）；009/014 若歸因給 binary 路則不進 AUPRO 帳；K5 重校 pen/lam 有下修風險；AUPRO>72 歷史 ceiling ~70.5 已被 v3 破，續漲空間本就薄 |
| AUROC | 85.05 | **+0.5 ~ +1.5**（→ ~85.5–86.5） | can/wallplugs goods flood 清理（K12/K5-Lambertian/K9）改善 ranking | 全域 kNN 項已刪（continuous-map 指標動不了）；can float 清理隨 V1-can 未知連動下修 |
| SegF1 | 63.89 | **+1.5 ~ +3.0**（→ ~65.5–67） | K7 翻轉 sheet_metal（28 peak-in-GT、0 FP 預算全空）/fabric（21）/rice（floor 機制已被同 specimen 證明）/walnuts（17＋009/014 若歸因成立）——信心高（訊號已在 float 裡）；K11 vial spill 抑制（新 owner，天花板 +0.5~1 cat）；K6 wallplugs 16 案部分回收 | vial 池 headroom 大但 K11 天花板有限；所有增益為毛額——K5/K6 強制 VarVal 重校常數取代 test-tuned 常數，前例（legitimacy audit）顯示此舉本身下修 headline，未動 cat 凍結 v3 常數為對沖；wallplugs 歷史最脆弱（信心低中） |
| ClassF1 | 83.65 | **+1.0 ~ +2.5**（→ ~84.5–86） | 偵測翻轉（fabric/rice/sheet_metal/walnuts，經 K7）；good FP 減少（can K12/wallplugs K5/K6）；K4 全域 kNN 輔助 | 相對 27 案 fabric/1 單項 ~+3 屬保守方向，可保留 |

**明確不承諾**：can 83/84 bad miss（sub-token 寫死）；wallplugs 51/67 零響應 camouflage；fj 4-variant specimen 顆粒粗（5/15 全漏 specimen 每救一個=4 張，量化跳動大）。

**最壞情境**（P1c 判 shift/shadow 增強不忠實 → 該類 gate 全降 unsupported）：只剩 photometric-gated 組件（K7 大部、K11、K12、K4）＋資料槓桿，約 SegF1 +1~+2、其餘各 +0.5~+1。

**單點故障**：VarVal 保真度（synthetic≠real 已在 vial gain 52→32 失敗過一次）——P1c 判準已改 normal-side 方向性並預留 margin，但「重現方向」≠「重現幅度」，所有 gate 門檻須留 buffer 且 regular 不退為硬條件。關鍵路徑=P1b 陰影合成品質＋P0/P1 串行前置；單 5090 可行，P2 軌可在 E3c 蒸餾續跑期間先行（binary 路為主，GPU 需求低）。

## 審查逐條 (KEEP/FIX 全文)

- **[FIX] 1. VarVal 變體增強校準集（含 E10a BLOCKER）** — 四個漏洞。(i) 合法性：E10a 的保真目標「rice floor 翻轉」是 test-GT 相依現象（004/014 在 shift_2/3 的偵測翻轉，讀的是 test bads 的偵測結果），can/wallplugs 條目也是在 test_public 上量到的方向性；作一次性預先登記的診斷可以，但 BLOCKER 框架會誘使對陰影/shift 合成參數反覆迭代「直到重現」——那等於把增強生成器擬合到 test 統計，VarVal 的合法性根基自毀。修正：保真目標限 normal-side 聚合方向性（good-image FP 趨勢），VarVal 建構預先登記、最多一輪修訂；rice 條目改用 held-normal floor 統計。(ii) can 判準注定失敗：specular/foil shift 已驗證不可合成（27 案審計 credibility ladder），「can shift-FP 單調上升」在合成 shift 上可預期地重現不了，照條文會無辜凍結整個 threshold 軌。改 per-cat/per-現象判準，can-specular 預先登記為不可重現、排除在 blocker 之外。(iii) 凍結分割不自洽：組件 2/3/5/6 的容量掃描、融合權重、E13 gate 全在 VarVal 上量——E10a 失敗時它們的 gate 同樣不可信，「只凍 7/8/9」是假隔離。正確降級 = 只保留光度增強（忠實）類 gate，shift/shadow 相依 gate 全降 unsupported。(iv) VarVal 實質是 27 案 P5 的基礎設施：繼承逐常數守門＋SAFE 回退，並把「VarVal 重校常數取代 test-tuned 常數 → 該 cat 分數預期下修」列為明文預算負項。
- **[FIX] 2. Branch C：低容量特徵 AE（logical 通路）** — 新軸成立、gate 設計（P99 vs SynLogical 2× 對比、增強訓練資料防變體殘差爆炸）良好，但三處必修。(i) 前提部分錯誤：CSV 驗證 walnuts 009 六變體全部 peak_in_gt=1、014 為 3/6——patch-local float 已在 GT 上開火，「局部 patch 落在正常流形內」對這兩個標本不成立；它們是 binary-chain 誤失（27 案 walnuts/1 的白撿 TP），必須先跑零成本 stage logging，C 不得把 009/014 入帳；C 的 walnuts 正當標的縮到 float 真正沉默的組成異常（如 004_shift_3 碎片、空殼未著火區）。(ii) vial 105/105 已偵測（CSV 驗證零漏檢）→ C 對 vial 無偵測/ClassF1 upside，價值只在 008/013 的定位（SegF1）與 AUPRO；融合權重在 SynLogical 上擬合 = vial 52→32 synth≠real 同族風險——vial 權重保守化＋vial-AUPRO 無回歸 gate（vial float 是強項）。(iii) 「一次性 test_public、預期 009/014 移動」需預先登記反事實決策：VarVal gate 過而命名影像未翻轉時，採用與否仍只依 VarVal，否則就是以 test-GT 命名影像做採用決策。
- **[FIX] 3. Branch C2：物件級 pooled-embedding kNN 塗回** — walnuts 版 = 27 案 walnuts/3 重包裝，必須繼承其審計修正：分數以 validation-加權「加法/max」項廣播，而非「塗回整顆 segment」硬替換（合併實例的假高分會注入大片 float FP）；加 walnuts-AUPRO 無回歸 gate（float 57.5>ISVL++ 53 是競爭強項）；接觸/重疊核需 watershed split＋QA。SAM3-偵測用途合法（已驗證）無衝突。009/014 與組件 2、7 三重入帳——stage logging 歸因後單一入帳。vial 帶 pooling：vial test 有 shift_1–4 共 4 個位移變體（CSV 驗證），帶必須以 product-ROI 相對座標定義，影像絕對座標帶會整排 FP；vial 零漏檢 → 收益路由是 SegF1/AUPRO 定位而非偵測；另預先登記註銷條件——train normals 液位自然變異若把 008 型含量異常收進 normal 帶流形，此路失效。bank-referenced 設計確實避開 per-image score normalization 負結果，該辯護成立。
- **[FIX] 4. 全域 embedding kNN 影像分數輔助** — 一個指標路由錯誤：AU-ROC 是 continuous-map 指標（本專案已驗證、審計 brief 明文），image-score 融合動不了它——「ClassF1+AUROC」的 AUROC 入帳刪除，只記 ClassF1；expected_gains 的 AUROC 項中「全域 kNN 撈回全漏影像」同步刪除。其餘設計（變體增強 bank 防 CLS 漂移、gate 不過即棄、明確零定位定位）正確，成本低，保留為 ClassF1-only 輔助。
- **[FIX] 5. V1：陰影/姿態/context 增強 membank 重建** — 拆成兩半。可信半：wallplugs/walnuts 的陰影/context 增強屬近朗伯場景，credibility ladder 支持；walnuts good FP 8 張全在 shift_1/3（CSV 驗證）目標真實；coreset 稀釋監控條款正確；補一條 membank 崩潰域警語（crop640+bank500+ROI 已知 native crash，bank 增長需留意）。不可信半：can——pose-jitter「真實 crop」仍只有 regular 光照下的 specular 外觀，2D 幾何變換無法合成物理位移下的 sheen/foil 變化（已驗證，正是 can/1 被降為「期望未知」的理由）；「can float-FP 清理 +1~2 信心中」無新證據不得覆寫「未知」。gate「can 12/12 shift-FP 單調性被打破」只能在 test_public goods 上量 = test-informed 採用 gate，必須改 VarVal-only（而 VarVal 又無法忠實表徵 can-shift → can 部分的 gate 實際不可評，期望誠實記未知）。此外 adoption matrix 的 can 條目（V1+E13c+V2）完全遺漏 27 案評為 can 最可靠槓桿的 can/3 幾何包絡抑制器（面積/aspect 特徵對外觀變化穩健）——can 條目重排為 can/3 先行、V2-binary 次之、V1-can 降為綁 can/3 的廉價實驗。
- **[KEEP] 6. D1：wallplugs 變體增強蒸餾頭** — 配方出處真實（E2d 111%/E3c 95%，非 bank-overlap 死路），且「蒸餾泛化超越 teacher」是能修暗變體動態範圍塌縮的正確機制。兩處微修：(i) teacher 在 51/67（CSV 驗證，非文中 49/67）誤失上本就零 float 響應——蒸餾不可能創造 camouflage 訊號，期望只記變體相依塌縮的恢復（16 個 peak-in-GT 可回收案＋動態範圍），不得暗示回收 51 案；(ii) 該配方在 wallplugs 未證過（fj 70% 還 pending），≥95% teacher gate、完整管線（hires 2.0+mask_fusion）下測、單 cat 回滾等防線如文保留。
- **[FIX] 7. Binarization v2：局部對比二值化** — 本體是已審 27 案 Tier-2 的重包裝（fabric/1+3、sheet_metal/1(+2/3)、rice/2、fj/1），指標路由正確（無 AUPRO 主張＋float hash 守門，修掉了 fabric/1 原版的唯一 metric-confusion 違規），FP 預算數字經 CSV 全部驗證屬實（rice goods 0/42、sheet_metal 0/24 且 28/31 miss peak-in-GT、fabric 2/72 共 48px、fj 1/20 共 9px、walnuts 17/30）。三處必修：(i) 順序——27 案 P2 substrate 決策（synthetic-val 導出 float-vs-distance）必須先行，否則同批 float-fires/binary-dead 誤失在基底更換後全部重校且被重複入帳；(ii) walnuts/wallplugs 的「小峰提升」：峰值是在 hires float 上量到的，base-res binary substrate 在同位置可能無幅度（wallplugs/2 審計修正）——種子必須改讀 hires fused-float 峰值；(iii) 若 sheet_metal 用延長/離心率豁免，繼承 sheet_metal/2 的 seam-band 排除＋方向 gate；相對 27 案為「取代」非「疊加」，每個 miss 池只入帳一次。rice floor 減法的 rice-only/binary-only/bit-identical 釘死條款正確。
- **[FIX] 8. V2：變體叢集條件門檻偏移（can/wallplugs）** — binary 版可留：叢集級粗粒度、held-normal-referenced、影像內容特徵（無檔名）、regular bit-identical——與 27 案 sheet_metal/3/rice/2 同族的合法 binary 範疇，vial 永不套用條款正確。float 試驗刪除或明記 unsupported：can 的 VarVal 無法忠實表徵 specular shift（已驗證）→「VarVal can-AUPRO ≥+1」這個 gate 在真正相關的分佈上量不到東西，形同虛設；且叢集偏移改變跨影像分數可比性，正是 vial oracle 21.9（跨影像原始分數可比性本身攜帶訊號）教訓的一般化——per-image 與 per-cluster 只差粒度。另 wallplugs 若做 float 試驗，gate 只在光度（忠實可合成）變體上有效力，shift 部分標注殘差缺口。
- **[KEEP] 9. 陰影邊界 FP gate** — 數字驗證：walnuts good FP 8 張確實全在 shift_1/3；但 wallplugs 的 good FP 有 4 張在 shift_1（CSV），範圍從「shift_2/3」修為「shift_1/2/3、shift_2 為主」。三處微修：(i) 去重——與 27 案 wallplugs/3(a) 觸邊背景 blob gate、walnuts/2 的 always-on 結構性防衛攻同一 FP 池，每 cat 擇一機制、FP-px 移除只入帳一次；(ii) walnuts goods 總 FP 僅 824px、wallplugs 1,236px——SegF1 影響微小，主要價值在 goods 的 float ranking（AUPRO/AUROC）與 ClassF1，期望帳目照此路由；(iii) 完整管線（gain 1.4+mask_fusion）下測、合成 defect-on-shadow-boundary 保留率 ≥90%、不碰 can（高頻 foil 不匹配）等條款全部正確保留。
- **[KEEP] 10. sheet_metal PoU/stitch 縫線修復** — 乾淨：診斷先行（縫位置 float 能量與 tile 邊界對齊才確認 artifact）、只修縫不動解析度（sheet_metal hires 69→51 反例教訓內建）、goods 0 FP 維持＋regular AUPRO 不退 gate。一條註記：診斷讀 003_regular test 影像屬「診斷可、調參禁」範疇（rice/1 Step1 前例），修復參數本身須由 tile 幾何/VarVal 導出，不得對該張影像調。
- **[FIX] experiment_plan：E-series 順序與 test_public 紀律** — 三個結構問題。(i) test 紀律在條文上就漏水：約 8–10 個「一次性」組件級 test_public 驗證＋E16 全量跑＋「任一 cat 任一 metric 退步→該 cat 回滾」= 以 test_public 結果做 per-cat 採用決策，與 legitimacy audit 定性的 adopt-on-test（SAM-adopt/can-scale 前科）同族，多次比較也侵蝕一次性原則。修正：採用/棄用只依 VarVal gate；組件級 test 峰視合併成軌級（T1/T2/T3 各一次）；E16 一次；回滾只允許預先登記的災難性門檻（例如單 cat 單 metric −2 以上），且一旦發生回滾，最終配置必須文件標注為 test_public-informed。(ii) 缺 27 案 Tier-1 廉價診斷：walnuts/1 stage logging（決定 009/014 歸屬，直接改寫 C/C2 的 walnuts 理由）、rice/1 Step1 階段歸因、fj/3＋wallplugs/1 的 hires A-map 重用稽核（可能讓組件 7 的 hires-seed 成本歸零）、vial/3 載具列稽核——全部零/極低成本，應排在 E10 之後、三軌之前。(iii) 排序倒置：Binarization v2（便宜、證據最強、只需光度增強＋合成缺陷的 VarVal 子集）不必等 shadow 合成成熟，可在 VarVal 光度部分就緒後先行；重機件（C/C2/V1/D1）後行。單 5090 可行但工期重，E10a 串行前置＋VarVal 陰影品質迭代是關鍵路徑，此評估誠實。
- **[FIX] expected_gains 帳目** — 四項修正。(i) 幽靈增益（最嚴重）：SegF1 最大單項「vial spill 抑制（bads mean fp 629px、fp>gt 39/105，信心中高）」——數字經 CSV 驗證屬實（628.7px、39/105），但 10 個組件無一負責 vial bad-image FP 抑制：組件 7 無 vial 條目、V2 明文 vial 永不套用、C/C2 是加訊號非抑制、adoption matrix 的 vial 條目只有 C+C2。修正：把 27 案 vial/1（標籤抑制，天花板 +0.5-1 cat）與 vial/2（氣泡場模式，保守版）加入 vial 條目，否則刪除該項——SegF1 估計相應下修為 +1.5~+3.0。(ii) 未入帳負項：E13 的 pen/lam/gain VarVal 重校 = 以 validation-derived 常數取代 test-tuned 常數，legitimacy-audit 前例顯示 headline 因此下修（SegF1 朝 43-55 方向）——所有增益是毛額，需明文扣除重校損失或凍結未動 cat 的 v3 常數。(iii) AUROC 項：全域 kNN 部分刪除（continuous-map 指標）；can float 清理部分隨 V1-can「期望未知」連動下修 → AUROC 誠實範圍 +0.5~+1.5。(iv) 事實小錯：wallplugs 零響應 51/67 非 49/67。ClassF1 估計（+1~+2.5）反而低於 27 案 fabric/1 單項的 ~+3 MEAN 審計值——保守方向，可保留。

### 審查總評

骨架判定 KEEP：additive per-cat opt-in、gate 不過即回滾、float-hash 自動檢查、binaries-from-base-maps 不動、VarVal 概念本身——這套設計吸收了先前 legitimacy audit 與 27 案審計的大部分教訓（無 AUPRO metric-confusion、per-image 適應鎖 binary、SAM3 只用於偵測、蒸餾用已證配方而非 bank-overlap）。但條文層面有六個系統性漏洞：(1) E10a 是最大的合法性風險而非保險——保真目標含 test-GT 相依現象（rice 004/014 偵測翻轉）與 test_public goods 行為，BLOCKER 框架誘使把 VarVal 增強參數迭代到重現 test 統計＝把整個計畫的合法面擬合到 test；且 can specular 判準已被驗證不可合成、注定失敗，會無辜凍結全部 threshold 軌；「凍 7/8/9 留 2/3/5/6」的分割不自洽，因為 2/3/5/6 的 gate 同樣全在 VarVal 上量。(2) SegF1 最大宣稱增益（vial spill 抑制）沒有任何組件負責——幽靈帳，需補 vial/1+vial/2 或刪項。(3) walnuts 009/014 被 C、C2、Binarization v2 三重入帳，而 CSV 證明 009 六變體 peak_in_gt=1（float 已開火）＝binary-chain 誤失，零成本 stage logging（27 案 Tier-1，v2 漏排）大概率讓 C/C2 的 walnuts 理由縮水。(4) test_public 紀律條文漏水：組件級一次性驗證 ×~10＋E16 回滾於 test 退步＝adopt-on-test 同族，需收斂為軌級峰視＋預先登記災難性回滾門檻。(5) can 軌建在唯一不可合成的增強（specular shift）上、gate 只能在 test goods 上量，卻遺漏已審最可靠的 can/3 幾何包絡。(6) 增益全是毛額——E13 強制的 VarVal 常數重校會把 test-tuned v3 常數換成誠實常數，前例顯示這本身下修 headline，未入預算。修正後誠實範圍：SegF1 +1.5~3.0、AUPRO +1.0~2.5、AUROC +0.5~1.5、ClassF1 +1.0~2.5。判定統計：KEEP 3（D1、陰影 gate、seam 修復）、FIX 9、KILL 0；條目級刪除三處——V2 的 can float 試驗、組件 4 的 AUROC 入帳、expected_gains 的 vial 幽靈項。執行順序改為：E10a（修正版、per-cat 判準）→ 27 案 Tier-1 廉價診斷（stage logging/重用稽核）→ Binarization v2（光度 VarVal 就緒即可）→ D1/V1-Lambertian → C/C2（依歸因結果縮圍）→ can 條目重組（can/3 先行）。單 5090 可行，關鍵路徑是 VarVal 陰影合成品質與 E10a 串行前置。