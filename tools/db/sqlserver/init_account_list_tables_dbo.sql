SET NOCOUNT ON;

BEGIN TRY
    IF OBJECT_ID(N'[dbo].[DMAccountLists]', N'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[DMAccountLists] (
            [AccountListID] BIGINT IDENTITY(1,1) NOT NULL,
            [Username] NVARCHAR(128) NOT NULL,
            [ListName] NVARCHAR(256) NOT NULL,
            [ConfigJson] NVARCHAR(MAX) NOT NULL,
            [UPDATE_DATE] DATETIME2(0) NOT NULL,
            [UPDATE_BY] NVARCHAR(128) NOT NULL,
            CONSTRAINT [PK_DMAccountLists] PRIMARY KEY CLUSTERED ([AccountListID] ASC),
            CONSTRAINT [CK_DMAccountLists_ConfigJson] CHECK (ISJSON([ConfigJson]) = 1)
        );
    END;

    IF OBJECT_ID(N'[dbo].[DMAccountListsArchive]', N'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[DMAccountListsArchive] (
            [AccountListID] BIGINT NOT NULL,
            [Username] NVARCHAR(128) NOT NULL,
            [ListName] NVARCHAR(256) NOT NULL,
            [ConfigJson] NVARCHAR(MAX) NOT NULL,
            [UPDATE_DATE] DATETIME2(0) NOT NULL,
            [UPDATE_BY] NVARCHAR(128) NOT NULL,
            [ARCHIVE_DATE] DATETIME2(0) NOT NULL,
            CONSTRAINT [CK_DMAccountListsArchive_ConfigJson] CHECK (ISJSON([ConfigJson]) = 1)
        );
    END;

    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE [name] = N'idx_dm_account_lists_username_name_date'
          AND [object_id] = OBJECT_ID(N'[dbo].[DMAccountLists]')
    )
    BEGIN
        CREATE INDEX [idx_dm_account_lists_username_name_date]
            ON [dbo].[DMAccountLists] ([Username] ASC, [ListName] ASC, [UPDATE_DATE] DESC);
    END;

    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE [name] = N'idx_dm_account_lists_username_date'
          AND [object_id] = OBJECT_ID(N'[dbo].[DMAccountLists]')
    )
    BEGIN
        CREATE INDEX [idx_dm_account_lists_username_date]
            ON [dbo].[DMAccountLists] ([Username] ASC, [UPDATE_DATE] DESC, [AccountListID] DESC);
    END;

    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE [name] = N'idx_dm_account_lists_archive_username_name_date'
          AND [object_id] = OBJECT_ID(N'[dbo].[DMAccountListsArchive]')
    )
    BEGIN
        CREATE INDEX [idx_dm_account_lists_archive_username_name_date]
            ON [dbo].[DMAccountListsArchive] ([Username] ASC, [ListName] ASC, [ARCHIVE_DATE] DESC);
    END;

    SELECT
        [TableName] = N'DMAccountLists',
        [Exists] = CASE WHEN OBJECT_ID(N'[dbo].[DMAccountLists]', N'U') IS NOT NULL THEN 1 ELSE 0 END,
        [RowCount] = CASE
            WHEN OBJECT_ID(N'[dbo].[DMAccountLists]', N'U') IS NULL THEN NULL
            ELSE (SELECT COUNT(*) FROM [dbo].[DMAccountLists])
        END;

    SELECT
        [TableName] = N'DMAccountListsArchive',
        [Exists] = CASE WHEN OBJECT_ID(N'[dbo].[DMAccountListsArchive]', N'U') IS NOT NULL THEN 1 ELSE 0 END,
        [RowCount] = CASE
            WHEN OBJECT_ID(N'[dbo].[DMAccountListsArchive]', N'U') IS NULL THEN NULL
            ELSE (SELECT COUNT(*) FROM [dbo].[DMAccountListsArchive])
        END;
END TRY
BEGIN CATCH
    THROW;
END CATCH;
