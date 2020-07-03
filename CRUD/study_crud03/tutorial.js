const { MongoClient } = require('mongodb');
const { ClientRequest } = require('http');

async function main() {
    const uri = "mongodb+srv://kimmjen:a197346852@cluster0-0zms8.mongodb.net/<dbname>?retryWrites=true&w=majority";
    const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true});
    try {
        await client.connect();
        // await listDatabases(client);
        // await createListing(
        //     client,
        //     {
        //         name: "Lovely Loft",
        //         summary: "A charming loft in Paris",
        //         bedrooms: 1,
        //         bathrooms: 1
        //     }
        // );
        //  CREATE
        // await createMultipleListings(client, [
        //     {
        //         name: "Infinite Views",
        //         summary: "Modern home with infinite views from the infinity pool",
        //         propoerty_type: "House",
        //         bedrooms: 5,
        //         bathrooms: 4.5,
        //         beds: 5
        //     },
        //     {
        //         name: "Private room in London",
        //         propoerty_type: "Apartment",
        //         bedrooms: 1,
        //         bathrooms: 1
        //     },
        //     {
        //         name: "Beautiful Beach House",
        //         summary: "Enjoy relaxed beach living in this house with a private beach",
        //         bedrooms: 4,
        //         bathrooms: 2.5,
        //         beds: 7,
        //         last_review: new Date()
        //     }
        // ])
        await findOneListingByName(client, "Infinite Views");
    } catch (e) {
        console.error(e);
    } finally {
        await client.close();
    }
}
main().catch(console.err);

// READING DOCUMENTS
async function findOneListingByName(clinet, nameOfListing) {
    const result = await client.db("sample_airbnb").collection("listingsAndReviews").findOne({ name: nameOfListing});
    
    if (result) {
        console.log(`Found a listing in the collection with the name '${nameOfListing}': `);
        console.log(result);
    } else {
        console.log(`No listings found with the name '${nameOfListing}'`);
    }
}

//  CREATE
async function createMultipleListings(client, newListings){
    const result = await client.db("sample_airbnb").collection("listingsAndReviews").insertMany(newListings);
    console.log(`${result.insertedCount} new Listing(s) created with the following id(s): `);
    console.log(result.insertedIds);
}

//  CREATE
async function createListing(client, newListing) {
    const result = await client.db("sample_airbnb").collection("listingsAndReviews").insertOne(newListing);
    // console.log(result);
    console.log(`New listing create with the following id: ${result.insertedId}`);
    // New listing create with the following id: 5eff0e065ba2b33230534787
}

async function listDatabases(client) {
    const databasesList = await client.db().admin().listDatabases();

    console.log("Databases: ");
    databasesList.databases.forEach(db => console.log(` - ${db.name}`));
};