# Data and interpretation

**Eurostat [DS-045409](https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/DS-045409)**: monthly value, mass and supplementary quantity. Belgium, Germany, France, the Netherlands, Italy and Spain; imports and exports; 2018–2025.

The snapshot was retrieved on **28 September 2026**. The 144 bundled raw files are unchanged. `data/source_manifest.json` records URLs, retrieval times and SHA-256 hashes.

| CN code | Engine capacity |
|---|---|
| 87113090 | Above 380–500 cc |
| 87114000 | Above 500–800 cc |
| 87115000 | Above 800 cc |

These cover part of combustion-motorcycle trade. Unit reference: [official CN 2025, page 378](https://statistiques.public.lu/dam-assets/fr/enquetes/enquetes-entreprises/intrastat-commerce-exterieur/codesnc2025-engl/cn8-auto-en-2025.pdf#page=378). No complete historical classification concordance is applied.

## Scope and units

- **Row:** month × reporter × partner × product × direction. Flow 1 = imports; 2 = exports. These are aggregates, not sales or individual shipments.
- **Measures:** EUR; mass in 100 kg converted to kg; positive supplementary quantities interpreted as items for these codes. EUR/kg and EUR/item are trade averages, not retail prices. Zero/missing denominators produce no ratio.
- **National totals:** use WORLD once. Check that individual partners plus unallocated QS/QV/QW reconcile with it for value, mass and quantity. Regional totals are not added.
- **Routes:** retain individual territories, including XS (Serbia), XK (Kosovo), XC (Ceuta) and XL (Melilla). Map UK/EL to GB/GR; reject collisions and unknown codes. [Eurostat geonomenclature](https://ec.europa.eu/eurostat/documents/3859598/21485884/KS-01-25-019-EN-N.pdf/d7a8cb5a-0d84-2924-f4db-11c0ddea6a73?download=true&t=1746538266757).
- **Missingness:** an absent month is not zero. An empty source flag means no annotation, not verified quality. Unallocated trade supports national totals but cannot identify a partner.

## Timing and limits

Detailed extra-EU data are usually published about 46 days after month-end; intra-EU data usually follow one month later. The forecast uses information through t−4 at the start of month t. This is a conservative scenario, not a reconstructed release calendar. [Eurostat publication guidance](https://ec.europa.eu/eurostat/documents/3859598/23178546/KS-01-26-012-EN-N.pdf/a502437f-c3d2-b1c8-df8b-95ed12e992a0?download=true), section 7.

Revisions, reporting thresholds, product mix and missing routes affect comparisons. There are no transaction records, retail sales, weather measurements or verified fraud outcomes. Imports and exports are analysed separately; adding reporters does not estimate unique European market size. See [Eurostat methodology](https://ec.europa.eu/eurostat/cache/metadata/en/ext_go_detail_sims.htm).

Source: Eurostat. The analysis filters, converts and models its data. Eurostat does not endorse these conclusions. [Reuse notice](https://ec.europa.eu/eurostat/help/copyright-notice).
