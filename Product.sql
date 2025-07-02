SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[Product] AS (select [$Outer].[StockItemID],
    [$Outer].[StockItemName],
    [$Outer].[SupplierID],
    [$Outer].[Size],
    [$Outer].[IsChillerStock],
    [$Outer].[TaxRate],
    [$Outer].[UnitPrice],
    [$Outer].[RecommendedRetailPrice],
    [$Outer].[TypicalWeightPerUnit],
    [$Inner].[StockGroupName]
from 
(
    select [$Outer].[StockItemID],
        [$Outer].[StockItemName],
        [$Outer].[SupplierID],
        [$Outer].[ColorID],
        [$Outer].[UnitPackageID],
        [$Outer].[OuterPackageID],
        [$Outer].[Brand],
        [$Outer].[Size],
        [$Outer].[LeadTimeDays],
        [$Outer].[QuantityPerOuter],
        [$Outer].[IsChillerStock],
        [$Outer].[Barcode],
        [$Outer].[TaxRate],
        [$Outer].[UnitPrice],
        [$Outer].[RecommendedRetailPrice],
        [$Outer].[TypicalWeightPerUnit],
        [$Outer].[MarketingComments],
        [$Outer].[InternalComments],
        [$Outer].[Photo],
        [$Outer].[CustomFields],
        [$Outer].[Tags],
        [$Outer].[SearchDetails],
        [$Outer].[LastEditedBy],
        [$Outer].[ValidFrom],
        [$Outer].[ValidTo],
        [$Inner].[StockGroupID]
    from [lh_FAIAD].[dbo].[ProductItem] as [$Outer]
    left outer join 
    (
        select [_].[StockItemStockGroupID] as [StockItemStockGroupID],
            [_].[StockItemID] as [StockItemID2],
            [_].[StockGroupID] as [StockGroupID],
            [_].[LastEditedBy] as [LastEditedBy2],
            [_].[LastEditedWhen] as [LastEditedWhen]
        from [lh_FAIAD].[dbo].[ProductItemGroup] as [_]
    ) as [$Inner] on ([$Outer].[StockItemID] = [$Inner].[StockItemID2] or [$Outer].[StockItemID] is null and [$Inner].[StockItemID2] is null)
) as [$Outer]
left outer join 
(
    select [_].[StockGroupID] as [StockGroupID2],
        [_].[StockGroupName] as [StockGroupName],
        [_].[LastEditedBy] as [LastEditedBy2],
        [_].[ValidFrom] as [ValidFrom2],
        [_].[ValidTo] as [ValidTo2]
    from [lh_FAIAD].[dbo].[ProductGroups] as [_]
) as [$Inner] on ([$Outer].[StockGroupID] = [$Inner].[StockGroupID2] or [$Outer].[StockGroupID] is null and [$Inner].[StockGroupID2] is null))
GO

