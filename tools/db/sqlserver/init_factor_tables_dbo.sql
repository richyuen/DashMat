SET NOCOUNT ON;
SET XACT_ABORT ON;

BEGIN TRY
    BEGIN TRAN;

    IF OBJECT_ID(N'[dbo].[FactorDefinitions]', N'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[FactorDefinitions] (
            [FactorName] NVARCHAR(128) NOT NULL,
            [LongComponent] NVARCHAR(4096) NOT NULL,
            [ShortComponent] NVARCHAR(4096) NULL,
            [Description] NVARCHAR(4096) NULL,
            [LongAggType] INT NOT NULL,
            [ShortAggType] INT NULL,
            [LongLag] INT NOT NULL,
            [OutputTransform] INT NOT NULL,
            [UPDATE_DATE] DATETIME2(0) NOT NULL,
            [UPDATE_BY] NVARCHAR(128) NOT NULL,
            CONSTRAINT [PK_FactorDefinitions] PRIMARY KEY CLUSTERED ([FactorName] ASC),
            CONSTRAINT [CK_FactorDefinitions_LongAggType] CHECK ([LongAggType] IN (1, 2, 3, 4, 5, 6, 7)),
            CONSTRAINT [CK_FactorDefinitions_ShortAggType] CHECK ([ShortAggType] IS NULL OR [ShortAggType] IN (1, 2, 3, 4, 5, 6, 7)),
            CONSTRAINT [CK_FactorDefinitions_LongLag] CHECK ([LongLag] >= 0),
            CONSTRAINT [CK_FactorDefinitions_OutputTransform] CHECK ([OutputTransform] IN (0, 1, 2))
        );
    END;

    IF OBJECT_ID(N'[dbo].[FactorDefinitionsArchive]', N'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[FactorDefinitionsArchive] (
            [FactorName] NVARCHAR(128) NOT NULL,
            [LongComponent] NVARCHAR(4096) NOT NULL,
            [ShortComponent] NVARCHAR(4096) NULL,
            [Description] NVARCHAR(4096) NULL,
            [LongAggType] INT NOT NULL,
            [ShortAggType] INT NULL,
            [LongLag] INT NOT NULL,
            [OutputTransform] INT NOT NULL,
            [UPDATE_DATE] DATETIME2(0) NOT NULL,
            [UPDATE_BY] NVARCHAR(128) NOT NULL,
            [ARCHIVE_DATE] DATETIME2(0) NOT NULL,
            CONSTRAINT [CK_FactorDefinitionsArchive_LongAggType] CHECK ([LongAggType] IN (1, 2, 3, 4, 5, 6, 7)),
            CONSTRAINT [CK_FactorDefinitionsArchive_ShortAggType] CHECK ([ShortAggType] IS NULL OR [ShortAggType] IN (1, 2, 3, 4, 5, 6, 7)),
            CONSTRAINT [CK_FactorDefinitionsArchive_LongLag] CHECK ([LongLag] >= 0),
            CONSTRAINT [CK_FactorDefinitionsArchive_OutputTransform] CHECK ([OutputTransform] IN (0, 1, 2))
        );
    END;

    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE [name] = N'idx_factor_defs_name'
          AND [object_id] = OBJECT_ID(N'[dbo].[FactorDefinitions]')
    )
    BEGIN
        CREATE INDEX [idx_factor_defs_name]
            ON [dbo].[FactorDefinitions] ([FactorName] ASC);
    END;

    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE [name] = N'idx_factor_defs_archive_name_date'
          AND [object_id] = OBJECT_ID(N'[dbo].[FactorDefinitionsArchive]')
    )
    BEGIN
        CREATE INDEX [idx_factor_defs_archive_name_date]
            ON [dbo].[FactorDefinitionsArchive] ([FactorName] ASC, [ARCHIVE_DATE] ASC);
    END;

    COMMIT TRAN;
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0
        ROLLBACK TRAN;
    THROW;
END CATCH;

SELECT
    [TableName] = N'FactorDefinitions',
    [Exists] = CASE WHEN OBJECT_ID(N'[dbo].[FactorDefinitions]', N'U') IS NOT NULL THEN 1 ELSE 0 END,
    [RowCount] = CASE
        WHEN OBJECT_ID(N'[dbo].[FactorDefinitions]', N'U') IS NULL THEN NULL
        ELSE (SELECT COUNT(*) FROM [dbo].[FactorDefinitions])
    END
UNION ALL
SELECT
    [TableName] = N'FactorDefinitionsArchive',
    [Exists] = CASE WHEN OBJECT_ID(N'[dbo].[FactorDefinitionsArchive]', N'U') IS NOT NULL THEN 1 ELSE 0 END,
    [RowCount] = CASE
        WHEN OBJECT_ID(N'[dbo].[FactorDefinitionsArchive]', N'U') IS NULL THEN NULL
        ELSE (SELECT COUNT(*) FROM [dbo].[FactorDefinitionsArchive])
    END;

SELECT
    [IndexName] = N'idx_factor_defs_name',
    [Exists] = CASE
        WHEN EXISTS (
            SELECT 1
            FROM sys.indexes
            WHERE [name] = N'idx_factor_defs_name'
              AND [object_id] = OBJECT_ID(N'[dbo].[FactorDefinitions]')
        ) THEN 1 ELSE 0
    END
UNION ALL
SELECT
    [IndexName] = N'idx_factor_defs_archive_name_date',
    [Exists] = CASE
        WHEN EXISTS (
            SELECT 1
            FROM sys.indexes
            WHERE [name] = N'idx_factor_defs_archive_name_date'
              AND [object_id] = OBJECT_ID(N'[dbo].[FactorDefinitionsArchive]')
        ) THEN 1 ELSE 0
    END;
