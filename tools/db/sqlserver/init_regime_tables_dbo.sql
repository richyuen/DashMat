SET NOCOUNT ON;
SET XACT_ABORT ON;

BEGIN TRY
    BEGIN TRAN;

    IF OBJECT_ID(N'[dbo].[RegimeDefinitions]', N'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[RegimeDefinitions] (
            [RegimeName] NVARCHAR(128) NOT NULL,
            [Description] NVARCHAR(4000) NULL,
            [MethodType] INT NOT NULL,
            [ConfigJson] NVARCHAR(MAX) NOT NULL,
            [UPDATE_DATE] DATETIME2(0) NOT NULL,
            [UPDATE_BY] NVARCHAR(128) NOT NULL,
            CONSTRAINT [PK_RegimeDefinitions] PRIMARY KEY CLUSTERED ([RegimeName] ASC),
            CONSTRAINT [CK_RegimeDefinitions_MethodType] CHECK ([MethodType] IN (1, 2, 3)),
            CONSTRAINT [CK_RegimeDefinitions_ConfigJson] CHECK (ISJSON([ConfigJson]) = 1)
        );
    END;

    IF OBJECT_ID(N'[dbo].[RegimeDefinitionsArchive]', N'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[RegimeDefinitionsArchive] (
            [RegimeName] NVARCHAR(128) NOT NULL,
            [Description] NVARCHAR(4000) NULL,
            [MethodType] INT NOT NULL,
            [ConfigJson] NVARCHAR(MAX) NOT NULL,
            [UPDATE_DATE] DATETIME2(0) NOT NULL,
            [UPDATE_BY] NVARCHAR(128) NOT NULL,
            [ARCHIVE_DATE] DATETIME2(0) NOT NULL,
            CONSTRAINT [CK_RegimeDefinitionsArchive_MethodType] CHECK ([MethodType] IN (1, 2, 3)),
            CONSTRAINT [CK_RegimeDefinitionsArchive_ConfigJson] CHECK (ISJSON([ConfigJson]) = 1)
        );
    END;

    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE [name] = N'idx_regime_defs_name'
          AND [object_id] = OBJECT_ID(N'[dbo].[RegimeDefinitions]')
    )
    BEGIN
        CREATE INDEX [idx_regime_defs_name]
            ON [dbo].[RegimeDefinitions] ([RegimeName] ASC);
    END;

    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE [name] = N'idx_regime_defs_archive_name_date'
          AND [object_id] = OBJECT_ID(N'[dbo].[RegimeDefinitionsArchive]')
    )
    BEGIN
        CREATE INDEX [idx_regime_defs_archive_name_date]
            ON [dbo].[RegimeDefinitionsArchive] ([RegimeName] ASC, [ARCHIVE_DATE] ASC);
    END;

    COMMIT TRAN;
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0
        ROLLBACK TRAN;
    THROW;
END CATCH;

SELECT
    [TableName] = N'RegimeDefinitions',
    [Exists] = CASE WHEN OBJECT_ID(N'[dbo].[RegimeDefinitions]', N'U') IS NOT NULL THEN 1 ELSE 0 END,
    [RowCount] = CASE
        WHEN OBJECT_ID(N'[dbo].[RegimeDefinitions]', N'U') IS NULL THEN NULL
        ELSE (SELECT COUNT(*) FROM [dbo].[RegimeDefinitions])
    END
UNION ALL
SELECT
    [TableName] = N'RegimeDefinitionsArchive',
    [Exists] = CASE WHEN OBJECT_ID(N'[dbo].[RegimeDefinitionsArchive]', N'U') IS NOT NULL THEN 1 ELSE 0 END,
    [RowCount] = CASE
        WHEN OBJECT_ID(N'[dbo].[RegimeDefinitionsArchive]', N'U') IS NULL THEN NULL
        ELSE (SELECT COUNT(*) FROM [dbo].[RegimeDefinitionsArchive])
    END;

SELECT
    [IndexName] = N'idx_regime_defs_name',
    [Exists] = CASE
        WHEN EXISTS (
            SELECT 1
            FROM sys.indexes
            WHERE [name] = N'idx_regime_defs_name'
              AND [object_id] = OBJECT_ID(N'[dbo].[RegimeDefinitions]')
        ) THEN 1 ELSE 0
    END
UNION ALL
SELECT
    [IndexName] = N'idx_regime_defs_archive_name_date',
    [Exists] = CASE
        WHEN EXISTS (
            SELECT 1
            FROM sys.indexes
            WHERE [name] = N'idx_regime_defs_archive_name_date'
              AND [object_id] = OBJECT_ID(N'[dbo].[RegimeDefinitionsArchive]')
        ) THEN 1 ELSE 0
    END;
