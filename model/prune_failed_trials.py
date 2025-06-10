import optuna
import argparse
import sqlite3
import torch # For checking torch.isnan if trial.value is a float

# 1. Set up argument parsing
parser = argparse.ArgumentParser(description="Optuna Study Trial Deletion Script (Direct SQLite Access)")
parser.add_argument('--study_name', type=str, required=True,
                    help="Name of the Optuna study (e.g., 'my_sr_upscaler_study')")
parser.add_argument('--db_name', type=str, default='db.sqlite3',
                    help="Name of the SQLite database file (e.g., 'db.sqlite3'). "
                         "This file will be accessed directly.")
args = parser.parse_args()

db_path = args.db_name
study_name = args.study_name

print(f"Attempting to manage study '{study_name}' directly from SQLite database '{db_path}'")

# Load the study to identify trial numbers (we still need this for trial states/values)
# Use optuna.load_study to get trial details, but we won't use its delete_trial method.
try:
    # We load it to get trial metadata, not to use its deletion methods.
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    print(f"Loaded study '{study_name}' with {len(study.trials)} trials for inspection.")
except Exception as e:
    print(f"Error loading study for inspection: {e}")
    print("Ensure the study name and database file are correct.")
    exit()

# Identify trials that returned 'inf' or are in a 'failed' state
trials_to_delete_numbers = []
for trial in study.trials:
    # Use optuna.trial.TrialState for compatibility with your version
    if trial.state == optuna.trial.TrialState.FAIL or \
       (trial.state == optuna.trial.TrialState.COMPLETE and \
        (trial.value is None or float(trial.value) == float('inf') or \
         float(trial.value) == float('-inf') or \
         (isinstance(trial.value, float) and trial.value != trial.value))): # Check for NaN
        trials_to_delete_numbers.append(trial.number)

if not trials_to_delete_numbers:
    print("No failed or 'inf' trials found to delete from the database.")
    exit()

print(f"Identified {len(trials_to_delete_numbers)} trials to delete: {trials_to_delete_numbers}")

# Connect directly to the SQLite database
conn = None
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the study_id for the given study_name
    cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    result = cursor.fetchone()
    if not result:
        print(f"Error: Study '{study_name}' not found in the database.")
        conn.close()
        exit()
    study_id = result[0]
    print(f"Found study_id: {study_id} for study_name: {study_name}")

    # Delete the identified trials
    for trial_number in trials_to_delete_numbers:
        try:
            # Delete from trials table (this was already working)
            cursor.execute("DELETE FROM trials WHERE study_id = ? AND number = ?", (study_id, trial_number))

            # Attempt to delete from trial_system_attrs and trial_user_attrs safely
            # These will only run if the tables exist.
            try:
                cursor.execute("DELETE FROM trial_system_attrs WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = ? AND number = ?)", (study_id, trial_number))
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    print(f"Skipping deletion from trial_system_attrs: Table does not exist. ({e})")
                else:
                    raise # Re-raise if it's another type of error
            except Exception as e: # Catch any other unexpected errors
                print(f"Unexpected error deleting from trial_system_attrs: {e}")
                
            try:
                cursor.execute("DELETE FROM trial_user_attrs WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = ? AND number = ?)", (study_id, trial_number))
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    print(f"Skipping deletion from trial_user_attrs: Table does not exist. ({e})")
                else:
                    raise # Re-raise if it's another type of error
            except Exception as e: # Catch any other unexpected errors
                print(f"Unexpected error deleting from trial_user_attrs: {e}")

            print(f"Successfully processed trial {trial_number} for deletion from database.")
        except sqlite3.Error as e:
            print(f"Error deleting trial {trial_number} from database: {e}")
            # Do not exit, try to delete other trials

    conn.commit() # Commit changes to the database
    print("\nDeletion process complete. Changes committed to database.")

except sqlite3.Error as e:
    print(f"Database error: {e}")
finally:
    if conn:
        conn.close()

# Verify deletion by reloading the study (optional)
try:
    study_after_deletion = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    print(f"Study '{study_name}' now has {len(study_after_deletion.trials)} trials after direct database deletion.")
except Exception as e:
    print(f"Could not reload study to verify: {e}")
    