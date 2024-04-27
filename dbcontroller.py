import os
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client


# create an abstract class called DBController
class DBController:
    def __init__(self):
        load_dotenv()
        url: str = os.getenv("DATABASE_URL")
        key: str = os.getenv("DATABASE_KEY")
        self.supabase: Client = create_client(url, key)
        self.table: str = "notes"

    def create(self, name: str) -> str:
        # Generate a UUID for the resource ID
        resource_id = str(uuid.uuid4())

        # Prepare the data to be inserted
        data = {
            "resourceid": resource_id,
            "name": name,
            "workstatus": "not_started"  # Assuming you want to set the default work status
        }

        # Insert the data into the table
        insert_query = self.supabase.table(self.table).insert([data])

        try:
            # Execute the insert query
            insert_query.execute()
            return resource_id
        except Exception as e:
            print("Error creating entry:", e)
            return None

    def has_id(self, id: str):
        try:
            res = self.supabase.table("notes").select("*").eq("resourceid", id).execute()
            return len(res.data) > 0
        except Exception as e:
            print("Error checking ID:", e)
            return False

    def set_context(self, id: str, context: str):
        if not self.has_id(id):
            return False

        # Update the context for the given resource_id
        update_query = self.supabase.table("notes").update({"context": context}).eq("resourceid", id)

        try:
            # Execute the update query
            update_query.execute()
            return True
        except Exception as e:
            print("Error updating context:", e)
            return False

    def set_style(self, id: str, style: str):
        if not self.has_id(id):
            return False

        # Update the style for the given resource_id
        update_query = self.supabase.table(self.table).update({"style": style}).eq("resourceid", id)

        try:
            # Execute the update query
            update_query.execute()
            return True
        except Exception as e:
            print("Error updating style:", e)
            return False

    def set_notes(self, id: str, notes: str):
        if not self.has_id(id):
            return False

        # Update the notes for the given resource_id
        update_query = self.supabase.table(self.table).update({"notes": notes}).eq("resourceid", id)

        try:
            # Execute the update query
            update_query.execute()
            return True
        except Exception as e:
            print("Error updating notes:", e)
            return False

    def set_work_status(self, id: str, status: str):
        if not self.has_id(id):
            return False

        # Update the work status for the given resource_id
        update_query = self.supabase.table(self.table).update({"workstatus": status}).eq("resourceid", id)

        try:
            # Execute the update query
            update_query.execute()
            return True
        except Exception as e:
            print("Error updating work status:", e)
            return False

    def get_context(self, id: str):
        if not self.has_id(id):
            return None

        # Fetch the context for the given resource_id
        select_query = self.supabase.table(self.table).select("context").eq("resourceid", id)

        try:
            # Execute the select query
            response = select_query.execute()
            return response.data
        except Exception as e:
            print("Error fetching context:", e)
            return None

    def get_style(self, id: str):
        if not self.has_id(id):
            return None

        # Fetch the style for the given resource_id
        select_query = self.supabase.table(self.table).select("style").eq("resourceid", id)

        try:
            # Execute the select query
            response = select_query.execute()
            style = response["data"][0]["style"]
            return style
        except Exception as e:
            print("Error fetching style:", e)
            return None

    def get_notes(self, id: str):
        if not self.has_id(id):
            return None

        # Fetch the notes for the given resource_id
        select_query = self.supabase.table(self.table).select("notes").eq("resourceid", id)

        try:
            # Execute the select query
            response = select_query.execute()
            notes = response["data"][0]["notes"]
            return notes
        except Exception as e:
            print("Error fetching notes:", e)
            return None

    def get_work_status(self, id: str):
        if not self.has_id(id):
            return None

        # Fetch the work status for the given resource_id
        select_query = self.supabase.table(self.table).select("workstatus").eq("resourceid", id)

        try:
            # Execute the select query
            response = select_query.execute()
            work_status = response["data"][0]["workstatus"]
            return work_status
        except Exception as e:
            print("Error fetching work status:", e)
            return None

    def set_not_started(self, id: str):
        return self.set_work_status(id, "not_started")

    def set_in_progress(self, id: str):
        return self.set_work_status(id, "in_progress")

    def set_done(self, id: str):
        return self.set_work_status(id, "done")

    def is_not_started(self, id: str):
        return self.get_work_status(id) == "not_started"

    def is_in_progress(self, id: str):
        return self.get_work_status(id) == "in_progress"

    def is_done(self, id: str):
        return self.get_work_status(id) == "done"
