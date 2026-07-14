WITH
params AS (
    SELECT DATE '2026-04-30' AS fecha_analisis
),

grupo_catalogo AS (
    SELECT * FROM (
        VALUES
            ('Grupo_2', 'positiva / cuentas_entrenamiento', 'positiva', 'cuentas_entrenamiento', 2),
            ('Grupo_3', 'negativa / cuentas_entrenamiento', 'negativa', 'cuentas_entrenamiento', 3),
            ('Grupo_5', 'ambos_signos / cuentas_entrenamiento', 'ambos_signos', 'cuentas_entrenamiento', 5)
    ) AS t(
        grupo,
        nombre_grupo,
        classification_account_type,
        gl_account_hist_behaviour_type,
        orden
    )
),

/* ============================================================
   Universo 1 AS-OF:
   último estado conocido de cada cross_key_id <= fecha_analisis.
   Evita mezclar estados históricos viejos para la misma cuenta-divisa.
   ============================================================ */
universo_1_asof AS (
    SELECT
        cross_key_id,
        classification_account_type,
        gl_account_hist_behaviour_type,
        label_cutoff_date
    FROM (
        SELECT
            TRIM(CAST(cross_key_id AS VARCHAR)) AS cross_key_id,
            classification_account_type,
            gl_account_hist_behaviour_type,
            CAST(cutoff_date AS DATE) AS label_cutoff_date,
            ROW_NUMBER() OVER (
                PARTITION BY TRIM(CAST(cross_key_id AS VARCHAR))
                ORDER BY CAST(cutoff_date AS DATE) DESC, load_date DESC
            ) AS rn
        FROM mx_master.t_mmfi_mac_historical_balances
        WHERE CAST(cutoff_date AS DATE) <= (SELECT fecha_analisis FROM params)
    )
    WHERE rn = 1
),

universo_1_entrenables AS (
    SELECT
        u.cross_key_id,
        g.grupo,
        g.nombre_grupo,
        g.orden,
        u.classification_account_type,
        u.gl_account_hist_behaviour_type,
        u.label_cutoff_date
    FROM universo_1_asof u
    INNER JOIN grupo_catalogo g
        ON u.classification_account_type = g.classification_account_type
       AND u.gl_account_hist_behaviour_type = g.gl_account_hist_behaviour_type
),

/* ============================================================
   Universo 2 Nivel 1:
   DETA contable vs aplicativo al 2026-04-30.
   ============================================================ */
deta_base AS (
    SELECT
        TRIM(CAST(clave_interfaz AS VARCHAR)) AS clave_interfaz,
        TRIM(CAST(cuenta AS VARCHAR)) AS cuenta,
        TRIM(CAST(centro AS VARCHAR)) AS centro,
        CAST(divisa_raw AS VARCHAR) AS divisa_raw,
        TRY_CAST(fecha AS DATE) AS fecha,
        CAST(saldo_aplicativo AS DECIMAL(28, 2)) AS saldo_aplicativo,
        CAST(saldo_contable AS DECIMAL(28, 2)) AS saldo_contable
    FROM deta_nivel1_base
    WHERE TRY_CAST(fecha AS DATE) IS NOT NULL
),

deta_lag AS (
    SELECT
        d.*,
        LAG(fecha, 1) OVER (
            PARTITION BY clave_interfaz, cuenta, centro, divisa_raw
            ORDER BY fecha ASC
        ) AS fecha_anterior,
        LAG(saldo_aplicativo, 1) OVER (
            PARTITION BY clave_interfaz, cuenta, centro, divisa_raw
            ORDER BY fecha ASC
        ) AS saldo_aplicativo_anterior,
        LAG(saldo_contable, 1) OVER (
            PARTITION BY clave_interfaz, cuenta, centro, divisa_raw
            ORDER BY fecha ASC
        ) AS saldo_contable_anterior
    FROM deta_base d
),

universo_2_key AS (
    SELECT
        clave_interfaz,
        cuenta,
        centro,
        CASE
            WHEN divisa_raw IS NULL OR TRIM(divisa_raw) = '' THEN 'MXP'
            ELSE TRIM(divisa_raw)
        END AS divisa,
        CONCAT(
            cuenta,
            CASE
                WHEN divisa_raw IS NULL OR TRIM(divisa_raw) = '' THEN 'MXP'
                ELSE TRIM(divisa_raw)
            END
        ) AS cross_key_id,
        fecha,
        fecha_anterior,
        CAST(
            (saldo_aplicativo - saldo_aplicativo_anterior)
            -
            (saldo_contable - saldo_contable_anterior)
            AS DECIMAL(28, 2)
        ) AS diferencia_estanco,
        CASE
            WHEN fecha_anterior IS NULL THEN 'RESIDUO'
            WHEN CAST(
                (saldo_aplicativo - saldo_aplicativo_anterior)
                -
                (saldo_contable - saldo_contable_anterior)
                AS DECIMAL(28, 2)
            ) = CAST(0 AS DECIMAL(28, 2))
                THEN 'CUADRE'
            WHEN CAST(
                (saldo_aplicativo - saldo_aplicativo_anterior)
                -
                (saldo_contable - saldo_contable_anterior)
                AS DECIMAL(28, 2)
            ) <> CAST(0 AS DECIMAL(28, 2))
                THEN 'DESCUADRE'
            ELSE 'RESIDUO'
        END AS nivel_1
    FROM deta_lag
    WHERE fecha = (SELECT fecha_analisis FROM params)
),

universo_2 AS (
    SELECT
        cross_key_id,
        MAX(cuenta) AS cuenta,
        MAX(divisa) AS divisa,
        CASE
            WHEN SUM(CASE WHEN nivel_1 = 'DESCUADRE' THEN 1 ELSE 0 END) > 0
                THEN 'DESCUADRE'
            WHEN SUM(CASE WHEN nivel_1 = 'CUADRE' THEN 1 ELSE 0 END) > 0
                THEN 'CUADRE'
            ELSE 'RESIDUO'
        END AS nivel_1,
        COUNT(*) AS n_llaves,
        COUNT(DISTINCT clave_interfaz) AS n_interfaces,
        COUNT(DISTINCT centro) AS n_centros,
        SUM(CASE WHEN nivel_1 = 'CUADRE' THEN 1 ELSE 0 END) AS n_llaves_cuadre,
        SUM(CASE WHEN nivel_1 = 'DESCUADRE' THEN 1 ELSE 0 END) AS n_llaves_descuadre,
        SUM(COALESCE(diferencia_estanco, CAST(0 AS DECIMAL(28, 2)))) AS total_diferencia_estanco
    FROM universo_2_key
    GROUP BY cross_key_id
),

matriz_asof AS (
    SELECT
        u1.grupo,
        u1.nombre_grupo,
        u1.orden,
        COUNT(DISTINCT CASE WHEN u2.nivel_1 = 'CUADRE' THEN u1.cross_key_id END) AS cuadre,
        COUNT(DISTINCT CASE WHEN u2.nivel_1 = 'DESCUADRE' THEN u1.cross_key_id END) AS descuadre,
        COUNT(DISTINCT CASE WHEN u2.cross_key_id IS NULL OR u2.nivel_1 = 'RESIDUO' THEN u1.cross_key_id END) AS residuo
    FROM universo_1_entrenables u1
    LEFT JOIN universo_2 u2
        ON u1.cross_key_id = u2.cross_key_id
    GROUP BY
        u1.grupo,
        u1.nombre_grupo,
        u1.orden
)

SELECT
    grupo,
    nombre_grupo,
    cuadre AS CUADRE,
    descuadre AS DESCUADRE,
    residuo AS RESIDUO
FROM matriz_asof
ORDER BY orden;
