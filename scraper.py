from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from datetime import datetime

def get_exact_train_options(source: str, destination: str, date: str) -> list[dict]:

    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    from datetime import datetime

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=100)  # Use modern headless mode with realistic delay
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/112.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()

        # Go to Ixigo train page
        page.goto("https://www.ixigo.com/trains", timeout=60000)
        page.wait_for_load_state("networkidle")

        # Source
        page.get_by_test_id("search-form-origin").locator("div").nth(1).click()
        page.get_by_test_id("search-form-origin").get_by_test_id("autocompleter-input").fill(source)
        all_stations_option = page.locator("div.bg-white li").filter(has_text=source).filter(has_text="All stations")
        (all_stations_option.first if all_stations_option.count() > 0 else 
         page.locator("div.bg-white li").filter(has_text=source).first).click()

        # Destination
        page.get_by_test_id("search-form-destination").locator("div").nth(1).click()
        page.get_by_test_id("search-form-destination").get_by_test_id("autocompleter-input").fill(destination)
        all_stations_option = page.locator("div.bg-white li").filter(has_text=destination).filter(has_text="All stations")
        (all_stations_option.first if all_stations_option.count() > 0 else 
         page.locator("div.bg-white li").filter(has_text=destination).first).click()


        

        # Search
        page.get_by_test_id("book-train-tickets").click()

        page.wait_for_url("**/search/result/train/**", timeout=10000)

        # Inject date into URL
        target_date = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = target_date.strftime("%d%m%Y")

        # Build final search URL
        final_url = page.url
        parts = final_url.split('/')
        parts[8] = formatted_date  # Replace the date part
        new_url = '/'.join(parts)

        print(f"Navigating to: {new_url}")
        page.goto(new_url)
        ###
        page.wait_for_selector("div.train-listing-rows", timeout=10000)

        # Scroll to load trains
        for _ in range(6):
            page.mouse.wheel(0, 1500)
            page.wait_for_timeout(500)

        train_blocks = page.locator("div.train-listing-row").all()
        train_data = []

        for block in train_blocks:
            try:
                train_number = block.locator("span.train-number").inner_text(timeout=1000)
            except PlaywrightTimeoutError:
                train_number = "N/A"

            try:
                train_name = block.locator("span.train-name").first.inner_text(timeout=1000)
            except PlaywrightTimeoutError:
                train_name = "N/A"

            try:
                departure = block.locator("div.left-wing.u-ib div.time").first.inner_text(timeout=1000)
            except PlaywrightTimeoutError:
                departure = "N/A"

            try:
                arrival = block.locator("div.right-wing.u-ib div.time").first.inner_text(timeout=1000)
            except PlaywrightTimeoutError:
                arrival = "N/A"

            try:
                duration = block.locator(
                    "div.train-duration > div > a > div > div.timeline-widget.u-ib > div > div:nth-child(2)"
                ).first.inner_text(timeout=1000)
            except PlaywrightTimeoutError:
                duration = "N/A"

            prices = {}
            availability = {}

            try:
                class_blocks = block.locator("div.train-class-item")
                for j in range(class_blocks.count()):
                    class_block = class_blocks.nth(j)

                    try:
                        class_name = class_block.locator("span.train-class").first.inner_text(timeout=500)
                    except:
                        class_name = f"Class_{j}"

                    # Click class tab
                    try:
                        class_block.click()
                        page.wait_for_timeout(1500)
                    except:
                        pass

                    # Dynamic wait for availability
                    try:
                        avail_box = block.locator("div.train-status-item").first
                        avail_status = avail_box.locator("div.avail-status").inner_text(timeout=3000)
                        availability[class_name] = avail_status
                    except:
                        availability[class_name] = "N/A"

                    # Dynamic wait for price
                    try:
                        price_span = class_block.locator("span.train-fare-available > div > span:nth-child(2)")
                        class_price = price_span.first.inner_text(timeout=3000)
                        prices[class_name] = class_price
                    except:
                        prices[class_name] = "N/A"

            except Exception:
                prices = {}
                availability = {}

            train_data.append({
                "number": train_number,
                "name": train_name,
                "departure": departure,
                "arrival": arrival,
                "duration": duration,
                "prices": prices,
                "Availability": availability,
                "Date" : date
            })

        context.close()
        browser.close()

    return train_data
