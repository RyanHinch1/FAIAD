SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[Reseller] AS (select [$Outer].[ResellerID] as [ResellerID],
    [$Outer].[ResellerName] as [ResellerName],
    [$Outer].[PostalCityID] as [PostalCityID],
    [$Outer].[PhoneNumber] as [PhoneNumber],
    [$Outer].[FaxNumber] as [FaxNumber],
    [$Outer].[WebsiteURL] as [WebsiteURL],
    [$Outer].[DeliveryAddressLine1] as [DeliveryAddressLine1],
    [$Outer].[DeliveryAddressLine2] as [DeliveryAddressLine2],
    [$Outer].[DeliveryPostalCode] as [DeliveryPostalCode],
    [$Outer].[PostalAddressLine1] as [PostalAddressLine1],
    [$Outer].[PostalAddressLine2] as [PostalAddressLine2],
    [$Outer].[PostalPostalCode] as [PostalPostalCode],
    [$Inner].[BuyingGroupName] as [ResellerCompany]
from [lh_FAIAD].[dbo].[Customers] as [$Outer]
inner join 
(
    select [_].[BuyingGroupID] as [BuyingGroupID2],
        [_].[BuyingGroupName] as [BuyingGroupName],
        [_].[LastEditedBy] as [LastEditedBy2],
        [_].[ValidFrom] as [ValidFrom2],
        [_].[ValidTo] as [ValidTo2]
    from [lh_FAIAD].[dbo].[BuyingGroups] as [_]
) as [$Inner] on ([$Outer].[BuyingGroupID] = [$Inner].[BuyingGroupID2] or [$Outer].[BuyingGroupID] is null and [$Inner].[BuyingGroupID2] is null))
GO

