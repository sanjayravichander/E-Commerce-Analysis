# E-Commerce Analysis

The purpose of this document is to outline the objectives, methodologies, and expected outcomes of an e-commerce product analysis project focused on a major Indian e-commerce platform.
This analysis aims to provide valuable insights into various aspects of product data collected from the platform.

# Objectives:
2.1. To analyze product trends and patterns within the Indian e-commerce market.
2.2. To identify popular product categories and subcategories.
2.3. To assess pricing strategies and pricing dynamics across different product segments.
2.4. To understand customer preferences and buying behaviors.
2.5. To evaluate the impact of product reviews and ratings on sales.
2.6. To identify opportunities for product assortment optimization and enhancement.

# Plots Followed in Tableau:

# Popular Product Categories and Brands:

a. Bar chart showing the top-selling product categories:
Drag the "Product Category" dimension to Columns.
Drag the "Sales Volume" measure to Rows.
Sort the bars in descending order by "Sales Volume."
b. Bar chart or treemap showing the market share of different brands within each category:

Create a treemap by dragging "Product Category" to Columns and "Brand" to Rows.
Place the "Sales Volume" measure on the Color shelf to represent market share.
Adjust the size of the treemap tiles based on "Sales Volume" to reflect the relative market share of brands within each category.

c. Stacked bar chart comparing the sales performance of top brands across different categories:
Create a stacked bar chart by placing "Product Category" on Columns and "Sales Volume" on Rows.
Place "Brand" on the Color shelf to stack the bars by brand.
Limit the number of brands shown or focus on top-selling brands for clarity.

# Pricing Trends and Discount Strategies:

a. Line chart illustrating the average price trends over time for select product categories:
Create a line chart with "Date" on Columns and "Average Price" on Rows.
Use a filter to select specific product categories for analysis.

b. Box plot or violin plot showing the distribution of prices for popular products within each category:
Create a box plot by dragging "Product Category" to Columns and "Price" to Rows.
Use the Box Plot chart type to visualize the distribution of prices within each category.

c. Scatter plot showing the relationship between price and sales volume, with trend lines indicating discount strategies:
Create a scatter plot with "Price" on the x-axis and "Sales Volume" on the y-axis.
Add trend lines to show the relationship between price and sales volume.
Optionally, color-code the points to represent different discount strategies.

# Seller Behavior and Performance:

a. Bubble chart or scatter plot showing seller performance metrics such as sales revenue versus customer satisfaction ratings:
Create a scatter plot with "Sales Revenue" on the x-axis and "Customer Satisfaction Ratings" on the y-axis.
Adjust the size of the bubbles based on the number of transactions or sales volume.

b. Bar chart or heatmap displaying the distribution of sellers based on key performance indicators like order fulfillment time and customer feedback ratings:
Create a heatmap by placing seller performance metrics on Rows and Columns.
Use color gradients to represent the distribution of sellers across different performance categories.

# Customer Preferences and Product Satisfaction:

a. Word cloud visualizing the most commonly used words in positive and negative product reviews:
Utilize text analysis techniques outside of Tableau to generate a word cloud based on customer reviews.

b. Sentiment analysis dashboard showcasing the overall sentiment score of customer reviews over time:
Calculate sentiment scores for customer reviews using text analysis tools.
Create a line chart with "Date" on Columns and "Sentiment Score" on Rows.

c. Heatmap illustrating customer satisfaction ratings for different product categories and brands:
Create a heatmap with "Product Category" on Rows, "Brand" on Columns, and "Customer Satisfaction Ratings" as the color.

# Correlations between Attributes:

a. Correlation matrix heatmap displaying the relationships between attributes such as price, rating, and brand:
Use Tableau's built-in correlation matrix feature to visualize the relationships between attributes.

b. Scatter plot matrix showing pairwise correlations between multiple attributes, with color-coding for different product categories:
Create a scatter plot matrix with each attribute plotted against others.
Color-code the points by product category to identify patterns and correlations.

c. Parallel coordinates plot visualizing the relationships between multiple attributes across different product categories:
Use Tableau's parallel coordinates plot to visualize the relationships between multiple attributes across different categories.

# Wordcloud Images 

![image](https://github.com/sanjayravichander/E-Commerce-Analysis/assets/86998084/29e3e145-459d-462f-b726-3d13105704da)

![image](https://github.com/sanjayravichander/E-Commerce-Analysis/assets/86998084/b3d085b4-653c-4886-a351-9338a105f1f9)

