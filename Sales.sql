SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[Sales] AS (select [$Outer].[InvoiceLineID] as [InvoiceLineID],
    [$Outer].[InvoiceID] as [InvoiceID],
    [$Outer].[StockItemID] as [StockItemID],
    [$Outer].[Quantity] as [Quantity],
    [$Outer].[UnitPrice] as [UnitPrice],
    [$Outer].[TaxRate] as [TaxRate],
    [$Outer].[TaxAmount] as [TaxAmount],
    [$Outer].[LineProfit] as [LineProfit],
    [$Outer].[ExtendedPrice] as [ExtendedPrice],
    [$Outer].[CustomerID] as [ResellerID],
    [$Outer].[SalespersonPersonID] as [SalespersonPersonID],
    [$Outer].[InvoiceDate] as [InvoiceDate],
    [$Outer].[t0_0] as [Sales Amount]
from 
(
    select [_].[InvoiceLineID] as [InvoiceLineID],
        [_].[InvoiceID] as [InvoiceID],
        [_].[StockItemID] as [StockItemID],
        [_].[Quantity] as [Quantity],
        [_].[UnitPrice] as [UnitPrice],
        [_].[TaxRate] as [TaxRate],
        [_].[TaxAmount] as [TaxAmount],
        [_].[LineProfit] as [LineProfit],
        [_].[ExtendedPrice] as [ExtendedPrice],
        [_].[CustomerID] as [CustomerID],
        [_].[SalespersonPersonID] as [SalespersonPersonID],
        [_].[InvoiceDate] as [InvoiceDate],
        [_].[ExtendedPrice] - [_].[TaxAmount] as [t0_0]
    from 
    (
        select [$Outer].[InvoiceLineID],
            [$Outer].[InvoiceID],
            [$Outer].[StockItemID],
            [$Outer].[Quantity],
            [$Outer].[UnitPrice],
            [$Outer].[TaxRate],
            [$Outer].[TaxAmount],
            [$Outer].[LineProfit],
            [$Outer].[ExtendedPrice],
            [$Inner].[CustomerID],
            [$Inner].[SalespersonPersonID],
            [$Inner].[InvoiceDate]
        from [lh_FAIAD].[dbo].[InvoiceLineItems] as [$Outer]
        inner join 
        (
            select [_].[InvoiceID] as [InvoiceID2],
                [_].[CustomerID] as [CustomerID],
                [_].[BillToResellerID] as [BillToResellerID],
                [_].[OrderID] as [OrderID],
                [_].[DeliveryMethodID] as [DeliveryMethodID],
                [_].[ContactPersonID] as [ContactPersonID],
                [_].[AccountsPersonID] as [AccountsPersonID],
                [_].[SalespersonPersonID] as [SalespersonPersonID],
                [_].[PackedByPersonID] as [PackedByPersonID],
                [_].[InvoiceDate] as [InvoiceDate],
                [_].[CustomerPurchaseOrderNumber] as [CustomerPurchaseOrderNumber],
                [_].[IsCreditNote] as [IsCreditNote],
                [_].[CreditNoteReason] as [CreditNoteReason],
                [_].[Comments] as [Comments],
                [_].[DeliveryInstructions] as [DeliveryInstructions],
                [_].[InternalComments] as [InternalComments],
                [_].[TotalDryItems] as [TotalDryItems],
                [_].[TotalChillerItems] as [TotalChillerItems],
                [_].[DeliveryRun] as [DeliveryRun],
                [_].[RunPosition] as [RunPosition],
                [_].[ReturnedDeliveryData] as [ReturnedDeliveryData],
                [_].[ConfirmedDeliveryTime] as [ConfirmedDeliveryTime],
                [_].[ConfirmedReceivedBy] as [ConfirmedReceivedBy],
                [_].[LastEditedBy] as [LastEditedBy2],
                [_].[LastEditedWhen] as [LastEditedWhen2]
            from [lh_FAIAD].[dbo].[Invoices] as [_]
        ) as [$Inner] on ([$Outer].[InvoiceID] = [$Inner].[InvoiceID2] or [$Outer].[InvoiceID] is null and [$Inner].[InvoiceID2] is null)
    ) as [_]
) as [$Outer]
where exists 
(
    select 1
    from 
    (
        select [ResellerID]
        from [lh_FAIAD].[dbo].[Reseller] as [$Table]
    ) as [$Inner]
    where [$Outer].[CustomerID] = [$Inner].[ResellerID] or [$Outer].[CustomerID] is null and [$Inner].[ResellerID] is null
))
GO

