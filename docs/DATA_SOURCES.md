# Source and units

**Eurostat [DS-045409](https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/DS-045409):** monthly trade value, mass and supplementary quantity. BE, DE, FR, NL, IT and ES; imports and exports; 2018–2025.

Snapshot retrieved **28 September 2026**. The 144 raw files (6.21 MB) are bundled. `data/source_manifest.json` records URLs, retrieval times and SHA-256 hashes.

| Product | Engine capacity |
|---|---|
| 87113090 | Above 380 to 500 cm³ |
| 87114000 | Above 500 to 800 cm³ |
| 87115000 | Above 800 cm³ |

These categories cover part of combustion-engine motorcycle trade.

## Interpretation

- **Row:** month × reporter × partner × product × direction. Flow 1 = imports; 2 = exports. Rows aggregate trade; they are not individual shipments.
- **Units:** value is in EUR; `mass_100kg × 100` gives kg. EUR/kg requires positive mass and is not a retail price. Supplementary quantity is retained without assuming a universal unit.
- **Flags:** empty means no annotation, not verified quality. Source IDs and update dates remain available.
- **Scope:** exclude overlapping totals, special codes (including XI) and self-partners. Map UK/EL to GB/GR and check collisions.
- **Gaps:** absent rows are not zeros. Complete country calendars can hide incomplete routes.
- **Comparability:** revised data, reporting thresholds and changing composition affect comparisons. No full historical code concordance is applied. See [Eurostat methodology](https://ec.europa.eu/eurostat/cache/metadata/en/ext_go_detail_sims.htm).

## Attribution

Source: Eurostat. Raw bytes are unchanged. This analysis filters, converts units, aggregates and estimates statistical models. Eurostat is not responsible for these transformations or conclusions and does not endorse the project. See its [reuse notice](https://ec.europa.eu/eurostat/help/copyright-notice).
