"""
Populate Real UO Sponsor Data
Based on University of Oregon's actual sponsorship agreements
"""

from sponsor_manager import add_sponsor, add_sponsor_history
from database_schema import init_database, reset_database

def populate_uo_sponsors():
    """Add real University of Oregon sponsors to the database."""

    # Reset database to clear sample data
    print("🗑️  Clearing sample data...")
    reset_database()

    print("\n📝 Adding real UO sponsors...\n")

    # 1. Nike - Exclusive Athletic Apparel Partner
    nike_id = add_sponsor(
        name="Nike",
        category="Athletic Apparel",
        relationship_type="current",
        is_exclusive=True,
        start_date="1990-01-01",
        sports_focus="All sports",
        contract_value=None,
        notes="Phil Knight (Nike founder) is a UO alumnus and major donor. Nike historically supplies apparel & equipment. Longstanding relationship with major re-ups; deal through 2028 reported in 2017. UO–Nike partnership dates to 1990s era. Source: KVAL"
    )
    add_sponsor_history(nike_id, "1990-01-01", "signed", "Initial partnership established")
    add_sponsor_history(nike_id, "2017-01-01", "renewed", "Major renewal - deal extended through 2028")

    # 2. Bigfoot Beverages (Pepsi distributor)
    bigfoot_id = add_sponsor(
        name="Bigfoot Beverages",
        category="Beverages",
        relationship_type="current",
        is_exclusive=True,
        start_date="2014-01-01",
        sports_focus="All campus",
        notes="Pepsi distributor. Handles campus pouring rights - vending, fountain sales and cooler space. Bigfoot was reported as UO pouring-rights distributor (2014 reporting); public records & later local reporting confirm continued commercial relationship and labor disputes in 2024–25. Source: Oregon Business"
    )
    add_sponsor_history(bigfoot_id, "2014-01-01", "signed", "Pouring rights agreement for campus beverage distribution")
    add_sponsor_history(bigfoot_id, "2024-01-01", "event", "Subject to labor/strike coverage")

    # 3. Gatorade
    gatorade_id = add_sponsor(
        name="Gatorade",
        category="Beverages",
        relationship_type="past",
        is_exclusive=False,
        start_date="2010-01-01",
        end_date="2020-07-31",
        sports_focus="All sports - on-field hydration",
        notes="Historically exclusive on-field drink provider. Past on-field hydration partner; contract expired July 31, 2020. Public records cite a Gatorade agreement that expired; check current pouring-rights docs for updates. Source: publicrecords.uoregon.edu"
    )
    add_sponsor_history(gatorade_id, "2020-07-31", "ended", "Contract expired")

    # 4. Bi-Mart - Charter Partner
    bimart_id = add_sponsor(
        name="Bi-Mart",
        category="Retail",
        relationship_type="current",
        is_exclusive=False,
        start_date="2024-08-05",
        sports_focus="All sports",
        notes="Charter Partner recognized by Oregon Sports Properties / Learfield. Named a Charter Partner in Learfield announcement (Aug 5, 2024). Source: Learfield"
    )
    add_sponsor_history(bimart_id, "2024-08-05", "signed", "Named Charter Partner by Oregon Sports Properties")

    # 5. Carl's Jr. - Charter Partner
    carls_id = add_sponsor(
        name="Carl's Jr.",
        category="Food & Dining",
        relationship_type="current",
        is_exclusive=False,
        start_date="2024-08-05",
        sports_focus="All sports",
        notes="Charter Partner announced Aug 5, 2024 by Oregon Sports Properties / Learfield. Source: Learfield"
    )
    add_sponsor_history(carls_id, "2024-08-05", "signed", "Named Charter Partner")

    # 6. Country Financial - Charter Partner
    country_id = add_sponsor(
        name="Country Financial",
        category="Financial Services",
        relationship_type="current",
        is_exclusive=False,
        start_date="2024-08-05",
        sports_focus="All sports",
        notes="Charter Partner announced Aug 5, 2024. Source: Learfield"
    )
    add_sponsor_history(country_id, "2024-08-05", "signed", "Named Charter Partner")

    # 7. McDonald's - Charter Partner
    mcdonalds_id = add_sponsor(
        name="McDonald's",
        category="Food & Dining",
        relationship_type="current",
        is_exclusive=False,
        start_date="2024-08-05",
        sports_focus="All sports",
        notes="Charter Partner announced Aug 5, 2024. Source: Learfield"
    )
    add_sponsor_history(mcdonalds_id, "2024-08-05", "signed", "Named Charter Partner")

    # 8. Oregon Community Credit Union (OCCU)
    occu_id = add_sponsor(
        name="Oregon Community Credit Union",
        category="Financial Services",
        relationship_type="current",
        is_exclusive=False,
        start_date="2024-08-05",
        sports_focus="All sports",
        notes="OCCU - Charter Partner announced Aug 5, 2024. Source: Learfield"
    )
    add_sponsor_history(occu_id, "2024-08-05", "signed", "Named Charter Partner")

    # 9. Toyota - Charter Partner
    toyota_id = add_sponsor(
        name="Toyota",
        category="Automotive",
        relationship_type="current",
        is_exclusive=False,
        start_date="2024-08-05",
        sports_focus="All sports",
        notes="Charter Partner announced Aug 5, 2024. Source: Learfield"
    )
    add_sponsor_history(toyota_id, "2024-08-05", "signed", "Named Charter Partner")

    # 10. Oregon Sports Properties / Learfield
    learfield_id = add_sponsor(
        name="Oregon Sports Properties",
        category="Media & Entertainment",
        relationship_type="current",
        is_exclusive=True,
        start_date="2024-01-01",
        sports_focus="All sports",
        notes="Learfield manages major multimedia and sponsorship agreements for Oregon Athletics. Learfield (Oregon Sports Properties) is the rights holder/manager for many sponsorships; contact them for partnership packages. Source: Learfield"
    )
    add_sponsor_history(learfield_id, "2024-01-01", "signed", "Multimedia rights agreement")

    # 11. Papé
    pape_id = add_sponsor(
        name="Papé",
        category="Automotive",
        relationship_type="current",
        is_exclusive=False,
        start_date="2015-01-01",
        sports_focus="Multiple facilities",
        notes="Facility naming / regional corporate partner. Multiple gifts including Papé Field naming; ongoing partnership. Papé family gifts & naming appear across UO projects. Source: DLR Group"
    )
    add_sponsor_history(pape_id, "2015-01-01", "signed", "Facility naming partnership established")

    # 12. Safeway / Albertsons
    safeway_id = add_sponsor(
        name="Safeway",
        category="Retail",
        relationship_type="current",
        is_exclusive=False,
        start_date="2020-01-01",
        sports_focus="All sports",
        notes="Regional grocery partner. Referenced in Oregon Sports Properties partner posts and campus partner messaging. Source: Facebook"
    )

    # 13. First Interstate Bank
    fib_id = add_sponsor(
        name="First Interstate Bank",
        category="Financial Services",
        relationship_type="current",
        is_exclusive=False,
        start_date="2020-01-01",
        sports_focus="All sports",
        notes="Mentioned as a partner in Oregon Sports Properties materials (active in regional college sponsorships). Source: Facebook"
    )

    # 14. PacificSource Health Plans
    pacific_id = add_sponsor(
        name="PacificSource Health Plans",
        category="Healthcare",
        relationship_type="current",
        is_exclusive=False,
        start_date="2020-01-01",
        sports_focus="All sports",
        notes="Cited by Oregon Sports Properties / partner social posts as a partner. Source: Facebook"
    )

    # 15. Aramark
    aramark_id = add_sponsor(
        name="Aramark",
        category="Food & Dining",
        relationship_type="current",
        is_exclusive=True,
        start_date="2018-01-01",
        sports_focus="Concessions",
        notes="Concessions / Food Service partner. Listed in UO public records as a concessions/vendor used by athletics. Concessions management is frequently a contracted sponsorship/service. Source: University of Oregon Public Records"
    )

    # 16. Collegiate Licensing Co.
    clc_id = add_sponsor(
        name="Collegiate Licensing Co.",
        category="Retail",
        relationship_type="current",
        is_exclusive=False,
        start_date="2015-01-01",
        sports_focus="Licensing",
        notes="Licensing / Trademark management. Listed on UO public records pages as trademark/licensing vendor. Source: University of Oregon Public Records"
    )

    # 17. Holland America Line
    holland_id = add_sponsor(
        name="Holland America Line",
        category="Travel & Hospitality",
        relationship_type="current",
        is_exclusive=False,
        start_date="2023-02-01",
        sports_focus="All sports",
        notes="Travel / Cruise Line partner. Signed as official sponsor in Feb 2023 with multi-year agreement. Source: PR Newswire"
    )
    add_sponsor_history(holland_id, "2023-02-01", "signed", "Multi-year agreement as official sponsor")

    print("\n✅ Real UO sponsor data populated successfully!")
    print("\n📊 Summary:")
    print("   - 15 Current Sponsors")
    print("   - 1 Past Sponsor (Gatorade - expired 2020)")
    print("   - Key Partners: Nike (exclusive apparel), Bigfoot Beverages (pouring rights), Learfield (multimedia rights)")
    print("\n🔍 Categories covered:")
    print("   • Athletic Apparel (Nike - EXCLUSIVE)")
    print("   • Beverages (Bigfoot - EXCLUSIVE pouring rights)")
    print("   • Financial Services (3 partners)")
    print("   • Food & Dining (3 partners)")
    print("   • Automotive (2 partners)")
    print("   • Healthcare (1 partner)")
    print("   • Retail (3 partners)")
    print("   • Media & Entertainment (Learfield - EXCLUSIVE)")
    print("   • Travel & Hospitality (1 partner)")
    print("\n💡 Test queries:")
    print("   - 'Can we propose Nike as a sponsor?' → NO (already exclusive)")
    print("   - 'Can we propose Adidas?' → NO (Nike exclusive in Athletic Apparel)")
    print("   - 'Can we propose Under Armour?' → NO (Nike exclusive)")
    print("   - 'Can we propose Coca-Cola?' → CONFLICT (Bigfoot has pouring rights)")
    print("   - 'Can we propose Delta Airlines?' → YES (Travel & Hospitality not exclusive)")
    print("   - 'Tell me about our Nike partnership' → Detailed info since 1990s")
    print("   - 'Who are our Charter Partners?' → Lists 6 Charter Partners from Aug 2024")
    print("\n🔧 Next steps:")
    print("   1. Restart databaseAccess.py: python3 Project/Parse/databaseAccess.py")
    print("   2. Test queries in your chat interface")
    print("   3. Check API: http://localhost:5001/api/sponsors")


if __name__ == "__main__":
    populate_uo_sponsors()
